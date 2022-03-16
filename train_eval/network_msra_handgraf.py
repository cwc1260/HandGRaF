'''
main model of HandGRaF-Net
author: wencan cheng
date: 24/01/2022
'''

import torch
import torch.nn as nn
import math
import numpy as np
from pointutil import Conv1d,  Conv2d, BiasConv1d, PointNetSetAbstraction, Mapping
import torch.nn.functional as F

graph = np.array([[1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32)

class PointNet_Plus(nn.Module):
    def __init__(self):
        super(PointNet_Plus, self).__init__()
        
        self.encoder_1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=64, in_channel=3, mlp=[32,32,128])
        
        self.encoder_2 = PointNetSetAbstraction(npoint=128, radius=0.3, nsample=64, in_channel=128, mlp=[64,64,256])

        self.encoder_3 = nn.Sequential(Conv1d(in_channels=256+3, out_channels=128, bn=True, bias=False),
                                       Conv1d(in_channels=128, out_channels=128, bn=True, bias=False),
                                       Conv1d(in_channels=128, out_channels=512, bn=True, bias=False),
                                       nn.MaxPool1d(128,stride=1))

        self.fold1 = nn.Sequential(BiasConv1d(bias_length=21, in_channels=512, out_channels=256, bn=True),
                                    BiasConv1d(bias_length=21, in_channels=256, out_channels=256, bn=True),
                                    BiasConv1d(bias_length=21, in_channels=256, out_channels=128, bn=True))
        self.regress_1 = nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1)
       
        self.graph_2 = nn.Parameter(torch.from_numpy(graph).cuda().unsqueeze(0).unsqueeze(1).repeat(1,128,1,1),requires_grad=True)
        self.fold_2 = Mapping(nsample=64, in_channel=128, latent_channel=128, mlp=[256, 256, 256], mlp2=[256, 256], radius=0.3)
        self.regress_2 = nn.Conv1d(in_channels=256, out_channels=3, kernel_size=1)

        self.graph_3 = nn.Parameter(torch.from_numpy(graph).cuda().unsqueeze(0).unsqueeze(1).repeat(1,256,1,1),requires_grad=True)
        self.fold_3 = Mapping(nsample=64, in_channel=128, latent_channel=256, mlp=[256, 256, 256], mlp2=[256, 256], radius=0.3)
        self.regress_3 = nn.Conv1d(in_channels=256, out_channels=3, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pc, feat):
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1

        pc1, feat1 = self.encoder_1(pc, feat)# B, 3, 512; B, 64, 512
        
        pc2, feat2 = self.encoder_2(pc1, feat1)# B, 3, 256; B, 128, 256
        
        code = self.encoder_3(torch.cat((pc2, feat2),1))# B, 3, 128; B, 256, 128
        
        code = code.expand(code.size(0),code.size(1), 21)

        # featat1 = torch.cat((skeleton, code), 1)
        featat1 = self.fold1(code)
        joints1 = self.regress_1(featat1)

        embed = torch.matmul(featat1.unsqueeze(-2), self.graph_2.transpose(2,3)).squeeze(-2)

        local2 = self.fold_2(joints1, pc1, embed, feat1) # B, 256, 21
        joints2 = self.regress_2(local2) # B, 3, 21
        joints2 = joints2 + joints1

        embed = torch.matmul(local2.unsqueeze(-2),self.graph_3.transpose(2,3)).squeeze(-2)

        local3 = self.fold_3(joints2, pc1, embed, feat1)
        joints3 = self.regress_3(local3) # B, 3, 21
        joints3 = joints3 + joints2

        joints3 = joints3.transpose(1,2).contiguous().view(-1,63)
        joints2 = joints2.transpose(1,2).contiguous().view(-1,63)
        joints1 = joints1.transpose(1,2).contiguous().view(-1,63)
        
        return joints1, joints2, joints3


from thop import profile, clever_format
if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((1,3,1024)).float().cuda()
    model = PointNet_Plus().cuda()
    # print(model)

    macs, params = profile(model, inputs=(input,input))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3fM" % (total))

