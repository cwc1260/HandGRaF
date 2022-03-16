import os
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
from transform import transform
import pcl
import scipy.io as sio


dataset_dir='../../Hand-Pointnet/data/cvpr15_MSRAHandGestureDB/'
save_dir = '../data/MSRA/msra_process_fullsubvplenpy/'
subject_names=['P0','P1','P2','P3','P4','P5','P6','P7','P8']
# subject_names = ['P0']
gesture_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']

JOINT_NUM = 21
SAMPLE_NUM = 1024

msra_valid = sio.loadmat('./msra_valid.mat')['msra_valid']

counter = 0
timer = 0
t = time.time()

for sub_idx in range(len(subject_names)):
    # try:
    #     os.makedirs(save_dir + subject_names[sub_idx])
    # except OSError:
    #     pass
    for ges_idx in range(len(gesture_names)):
        gesture_dir = dataset_dir + subject_names[sub_idx] + '/' + gesture_names[ges_idx]
        save_gesture_dir = save_dir + subject_names[sub_idx] + '/' + gesture_names[ges_idx]
        if not os.path.exists(save_gesture_dir):
            os.makedirs(save_gesture_dir)
        print(save_gesture_dir)
        depth_files = os.listdir(gesture_dir)
        depth_files_bin = []
        for di in depth_files:
            # t = os.path.splitext(di)[1]
            if os.path.splitext(di)[1] == '.bin':
                depth_files_bin.append(di)
        depth_files_bin = sorted(depth_files_bin)

        fileId = open(gesture_dir+'/joint.txt', 'r')

        frame_num = int(fileId.readline())

        Point_Cloud_FPS = np.zeros((frame_num, SAMPLE_NUM, 6))
        Volume_rotate = np.zeros((frame_num, 3,3))
        Volume_length = np.zeros((frame_num, 1))
        Volume_offset = np.zeros((frame_num, 3))
        Volume_GT_XYZ = np.zeros((frame_num, JOINT_NUM,3))

        A = np.array(fileId.read().split(), dtype=np.float32).reshape((-1, 21, 3))

        valid = msra_valid[sub_idx, ges_idx]

        for frm_idx in range(len(depth_files_bin)):

            if valid[frm_idx] != 1:
                continue

            img_width, img_height, bb_left, bb_top, bb_right, bb_bottom = np.fromfile(gesture_dir + '/' + '%06d' % frm_idx + '_depth.bin',dtype=np.int32)[:6]
            im = np.fromfile(gesture_dir + '/' + '%06d' % frm_idx + '_depth.bin',dtype=np.float32)[6:]
            bb_width = bb_right - bb_left
            bb_height = bb_bottom - bb_top

            hand_depth = im.reshape((bb_height, bb_width))

            hand_points_sampled, jnt_xyz, rot, offset, length = transform(img_width, img_height, bb_width, bb_height, bb_left, bb_top, hand_depth, A[frm_idx])

            Point_Cloud_FPS[frm_idx] = hand_points_sampled
            Volume_GT_XYZ[frm_idx] = jnt_xyz
            Volume_length[frm_idx] = length
            Volume_rotate[frm_idx] = rot
            Volume_offset[frm_idx] = offset

            counter = counter + 1

        sio.savemat(save_gesture_dir + '/Point_Cloud_FPS.mat', {'Point_Cloud_FPS': Point_Cloud_FPS})
        sio.savemat(save_gesture_dir + '/Volume_length.mat', {'Volume_length': Volume_length})
        sio.savemat(save_gesture_dir + '/Volume_GT_XYZ.mat', {'Volume_GT_XYZ': Volume_GT_XYZ})
        sio.savemat(save_gesture_dir + '/Volume_offset.mat', {'Volume_offset': Volume_offset})
        sio.savemat(save_gesture_dir + '/Volume_rotate.mat', {'Volume_rotate': Volume_rotate})
        sio.savemat(save_gesture_dir + '/valid.mat', {'valid': valid})

timer += time.time() - t

print(timer / counter)
