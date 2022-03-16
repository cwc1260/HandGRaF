'''
evaluation
'''
import argparse
import os
import random
import progressbar
import time
import logging
import pdb
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from thop import profile, clever_format
import importlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

from dataset_msra import HandPointDataset
from dataset_msra import subject_names
from dataset_msra import gesture_names
from network_msra_handgraf import PointNet_Plus

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python eval.py
parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default = 21,  help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 3,  help='number of input point features')

parser.add_argument('--save_root_dir', type=str, default='../results',  help='output folder')
parser.add_argument('--test_index', type=int, default = 0,  help='test index for cross validation, range: 0~8')
parser.add_argument('--model', type=str, default = 'best_model.pth',  help='model name for training resume')
parser.add_argument('--test_path', type=str, default = '../data/MSRA/msra_process_purepy',  help='model name for training resume')

parser.add_argument('--model_name', type=str, default = 'msra_handgraf',  help='')
parser.add_argument('--gpu', type=str, default = '0',  help='gpu')


opt = parser.parse_args()
print (opt)

module = importlib.import_module('network_'+opt.model_name)

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = os.path.join(opt.save_root_dir, opt.model_name+'_adamw_rotaug', subject_names[opt.test_index])


# 1. Load data                                         
test_data = HandPointDataset(root_path=opt.test_path, opt=opt, train = False)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers), pin_memory=False)
                                          
print('#Test data:', len(test_data))
print (opt)

# 2. Define model, loss
netR = getattr(module, 'PointNet_Plus')()

if opt.ngpu > 1:
    netR.netR_1 = torch.nn.DataParallel(netR.netR_1, range(opt.ngpu))
    netR.netR_2 = torch.nn.DataParallel(netR.netR_2, range(opt.ngpu))
    netR.netR_3 = torch.nn.DataParallel(netR.netR_3, range(opt.ngpu))
if opt.model != '':
    netR.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
    
netR.cuda()
print(netR)

criterion = nn.MSELoss(size_average=True).cuda()

# 3. evaluation
torch.cuda.synchronize()

netR.eval()
test_mse = 0.0
test_wld_err = 0.0
test_wld_err_fold1 = 0.0
test_class_err = torch.zeros(21, 1).cuda()
timer = 0

saved_points = []
saved_gt = []
saved_fold1 = []
saved_final = []
saved_error = []
saved_length = []

dump_input = torch.randn((1,3,1024)).float().cuda()
traced_netR = torch.jit.trace(netR, (dump_input, dump_input))


for i, data in enumerate(tqdm(test_dataloader, 0)):
	torch.cuda.synchronize()
	# 3.1 load inputs and targets
	with torch.no_grad():
		points, volume_length, gt_pca, gt_xyz = data
		# gt_pca = Variable(gt_pca, volatile=True).cuda()
		points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()
		# points[:,:,:3] = torch.matmul(points[:,:,:3], rot.view(-1,3,3).transpose(1,2)) 
		# gt_xyz = torch.matmul(gt_xyz.view(-1, 21, 3), rot.view(-1,3,3).transpose(1,2)).view(-1, 63)
		# permutation = torch.randperm(points.size(1))
		# points = points[:,permutation,:]
		# 3.2.2 compute output
		t = time.time()
		fold1, fold2, estimation = traced_netR(points[:,:,:3].transpose(1,2), points[:,:,:3].transpose(1,2))
		timer += time.time() - t

		loss = (criterion(estimation, gt_xyz)*2+criterion(fold1, gt_xyz)+criterion(fold2, gt_xyz))*63
	torch.cuda.synchronize()
	test_mse = test_mse + loss.item()*len(points)

	# 3.3 compute error in world cs
	# wrist: [0], index_R: [1], index_T: [4], middle_R: [5], middle_T: [8], ring_R: [9], ring_T: [12], little_R: [13], little_T: [16], thumb_R: [17], thumb_T: [20] 
	outputs_xyz = estimation
	diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
	diff_sum = torch.sum(diff,2)
	diff_sum_sqrt = torch.sqrt(diff_sum)
	diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
	diff_mean_wld = torch.mul(diff_mean,volume_length)

	diff_mean_class = torch.mul(diff_sum_sqrt, volume_length)
	diff_mean_class = torch.sum(diff_mean_class,0).view(-1,1)

	test_wld_err = test_wld_err + diff_mean_wld.sum()
	test_class_err = test_class_err + diff_mean_class	

	# fold1
	outputs_xyz = fold1
	diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
	diff_sum = torch.sum(diff,2)
	diff_sum_sqrt = torch.sqrt(diff_sum)
	diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
	diff_mean_wld = torch.mul(diff_mean,volume_length)

	test_wld_err_fold1 = test_wld_err_fold1 + diff_mean_wld.sum()

# time taken
torch.cuda.synchronize()
# timer = time.time() - timer
timer = timer / len(test_data)
print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

# print mse
test_wld_err = test_wld_err / len(test_data)
print('average estimation error in world coordinate system: %f (mm)' %(test_wld_err))
test_wld_err_fold1 = test_wld_err_fold1/ len(test_data)
print('average fold1 error in world coordinate system: %f (mm)' %(test_wld_err_fold1))

test_class_err = test_class_err / len(test_data)
print("wrist error: ", test_class_err[0])
print("index_R: ", test_class_err[1])
print("index_T: ", test_class_err[4])
print("middle_R: ", test_class_err[5])
print("middle_T: ", test_class_err[8])
print("ring_R: ", test_class_err[9])
print("ring_T: ", test_class_err[12])
print("little_R: ", test_class_err[13])
print("little_T: ", test_class_err[16])
print("thumb_R: ", test_class_err[17])
print("thumb_T: ",test_class_err[20])
