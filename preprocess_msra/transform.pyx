cimport numpy as np
cimport cython
# from pcl cimport pcl_defs as cpp
# from pcl cimport pcl_features as pcl_ftr
import numpy as np
import pcl

@cython.boundscheck(False)
@cython.wraparound(False)
def transform(int img_width, int img_height, int bb_width, int bb_height, int bb_left, int bb_top, np.ndarray[float, ndim=2] hand_depth, np.ndarray[float, ndim=2] jnt_xyz):
    cdef float fFocal_MSRA_ = 1.0/241.42
    cdef int SAMPLE_NUM = 1024
    cdef int KNN_NUM = 40
    cdef np.ndarray[float, ndim=2] hand_3d = np.zeros((bb_width * bb_height, 3), dtype=np.float32)
    cdef np.ndarray[long, ndim=2] knn_idx = np.zeros((SAMPLE_NUM, KNN_NUM), dtype=int)
    cdef np.ndarray[float, ndim=2] knn_pts = np.zeros((KNN_NUM, 3), dtype=np.float32)
    cdef np.ndarray[float, ndim=2] normal = np.zeros((SAMPLE_NUM, 3), dtype=np.float32)
    cdef int val_num
    cdef int ii
    cdef int jj
    cdef int idx
    cdef int valid_idx = 0
    cdef float x
    cdef float y
    cdef float z
    cdef hand_points_sampled
    cdef hand_points_val
    cdef offset
    cdef dist

    # for ii in range(bb_height):
    #     for jj in range(bb_width):
    #         idx = (jj - 1) * bb_height + ii
    #         x = -(img_width / 2 - (jj + bb_left - 1)) * hand_depth[ii, jj] / fFocal_MSRA_
    #         if x == 0:
    #             continue
    #         y = (img_height / 2 - (ii + bb_top - 1)) * hand_depth[ii, jj] / fFocal_MSRA_
    #         if y == 0:
    #             continue
    #         z = hand_depth[ii, jj]
    #         if z == 0:
    #             continue
    #         hand_3d[valid_idx] = [x, y, z]
    #         valid_idx =+ 1
    X, Y = np.meshgrid(range(bb_width), range(bb_height))
    hand_3d[:, 2] = hand_depth.reshape(-1)
    hand_3d[:, 0] = (((X+bb_left-1) - img_width*0.5)*(hand_depth*fFocal_MSRA_)).reshape(-1)
    hand_3d[:, 1] = ((img_height*0.5 - (Y+bb_top-1))*(hand_depth*fFocal_MSRA_)).reshape(-1)

    mask = hand_3d[:,0] != 0
    mask = mask - hand_3d[:,1] != 0
    mask = mask - hand_3d[:,2] != 0
    val_num = np.sum(mask.astype(int))
    hand_points_val = hand_3d[mask]

    # offset = np.mean(hand_points_val, 0) / 275.0
    # hand_points_val = hand_points_val / 275.0  - offset
    offset = np.mean(hand_points_val, 0)
    hand_points_val = hand_points_val  - offset
    H = np.dot(hand_points_val.T, hand_points_val)

    rand_ind = np.random.choice(val_num, SAMPLE_NUM)

    hand_points_sampled = hand_points_val[rand_ind]

    u, s, v = np.linalg.svd(H)
    sort = s.argsort()[::-1]
    u = u[:, sort]

    if u[1, 0] < 0:
        u[:, 0] = -u[:, 0]
    if u[2, 2] < 0:
        u[:, 2] = -u[:, 2]
    u[:, 1] = np.cross(u[:, 2], u[:, 0])

    # dist = hand_points_sampled[:, np.newaxis, :] - hand_points_val[np.newaxis, :, :]
    # dist = np.sum(dist, -1)

    # knn_idx = np.argsort(dist, -1)[:, :KNN_NUM]
    # for i in range(SAMPLE_NUM):
    #     knn_pts = hand_points_val[knn_idx[i]]
    #     H_k = np.dot(knn_pts.T, knn_pts)
    #     u_k, s_k, _ = np.linalg.svd(H_k)
    #     normal[i] = u_k[:, 2]


    hand_point = pcl.PointCloud(hand_points_sampled)
    # normalEstimation = pcl.NormalEstimation()
    # normalEstimation.me.setInputCloud(hand_point)
    # normalEstimation.me.setKSearch(30)
    # normalEstimation.me.setSearchSurface(pcl.PointCloud(hand_points_sampled))
    # normalEstimation = NormalEstimation()
    # cdef pcl_ftr.NormalEstimation_t *cNormalEstimation = <pcl_ftr.NormalEstimation_t *>normalEstimation.me
    # cNormalEstimation.setInputCloud(hand_point)
    # cNormalEstimation.setSearchSurface(pcl.PointCloud(hand_points_sampled))
    feature = hand_point.make_NormalEstimation()
    feature.set_KSearch(30)
    feature.set_SearchSurface(pcl.PointCloud(hand_points_val))
    feature.set_ViewPoint(-offset[0], -offset[1], -offset[2])
    normals = feature.compute().to_array()
    normals = normals[:,:3]
    # print(normals.shape)
    # normals[normals[:,2] > 0] = 0-normals[normals[:,2] > 0]
    normals = np.matmul(normals[:,:3], u)

    hand_points_sampled = np.matmul(hand_points_sampled, u)

    scale = 1.2
    bb3d_x_len = scale*(max(hand_points_sampled[:,0])-min(hand_points_sampled[:,0]));

    hand_points_sampled = hand_points_sampled /bb3d_x_len
    jnt_xyz[...,2] = -jnt_xyz[...,2]
    jnt_xyz = jnt_xyz - offset
    jnt_xyz = np.matmul(jnt_xyz, u)
    jnt_xyz = jnt_xyz / bb3d_x_len
    hand_points_sampled = np.concatenate((hand_points_sampled, normals), axis=1)
    return hand_points_sampled, jnt_xyz, u, offset, bb3d_x_len #275.0