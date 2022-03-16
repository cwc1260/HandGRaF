import torch 
import torch.nn as nn 

import torch.nn.functional as F
from time import time
import numpy as np
from pointnet2 import pointnet2_utils

LEAKY_RATE = 0.1
use_bn = False


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        # new_xyz: sampled points position data, [B, N, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def joint2offset(joints, points, theta):

    device = joints.device
    B = joints.size(0)
    J = joints.size(-1)
    N = points.size(-1)
    joints_feature = joints.view(B,-1,1).repeat(1,1,N) #B, Jx3, N
    points_repeat = points.repeat(1, J, 1) #B, Jx3, N
    offset = joints_feature - points_repeat
    offset = offset.view(B,J,3,N)
    dist = torch.sqrt(torch.sum(torch.pow(offset,2),dim=2)+1e-8) #B, J, N
    offset_norm = (offset / (dist.unsqueeze(2))) #B, J, 3, N
    heatmap = theta - dist
    mask = heatmap.ge(0).float() #B, J, N
    offset_norm_mask = (offset_norm*mask.unsqueeze(2)).view(B,-1,N)
    heatmap_mask = heatmap * mask.float()

    return torch.cat((offset_norm_mask,heatmap_mask),dim=1)

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn, bias=True):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class BiasConv1d(nn.Module):
    def __init__(self, bias_length, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(BiasConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.randn((out_channels, bias_length)),requires_grad=True)
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)+self.bias.unsqueeze(0).repeat(x.size(0),1,1)))
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, mlp2=None, group_all=False, use_fps=True,
                 return_fps=False, use_xyz=True, use_act=True, act=F.relu, mean_aggr=False, use_instance_norm=False, bn=True, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.use_act = use_act
        self.mean_aggr = mean_aggr
        # self.act = act
        self.act = nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.use_fps = use_fps
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.knn = knn
        last_channel = (in_channel + 3) if use_xyz else in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            if bn:
                if use_instance_norm:
                    self.mlp_bns.append(nn.InstanceNorm2d(out_channel, affine=True))
                else:
                    self.mlp_bns.append(nn.BatchNorm2d(out_channel))

            last_channel = out_channel

        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    if use_instance_norm:
                        self.mlp2_bns.append(nn.InstanceNorm1d(out_channel, affine=True))
                    else:
                        self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel
        if knn is False:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, self.use_xyz)
        self.return_fps = return_fps

    def forward(self, xyz, points, new_xyz = None, fps_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz = xyz.contiguous()
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        points = points.contiguous()

        if new_xyz == None:
            if (self.group_all == False) and (self.npoint != -1) and self.use_fps:
                if fps_idx == None:
                    fps_idx = pointnet2_utils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
                new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx)  # [B, C, N]
            elif self.use_fps is False:
                # fps_idx = torch.arange(self.npoint,dtype=torch.int).view(1, -1).repeat(xyz.size(0),1)
                # new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx)
                new_xyz = xyz[...,:self.npoint]
            else:
                new_xyz = xyz

        if self.knn:
            sqrdists = square_distance(new_xyz.transpose(2, 1).contiguous(), xyz_t)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz_t, knn_idx)
            direction_xyz = neighbor_xyz - new_xyz.transpose(1,2).view(B, self.npoint, 1, C)

            grouped_points = index_points_group(points.transpose(1,2), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([direction_xyz, grouped_points], dim = -1)
            new_points = new_points.permute(0, 3, 1, 2)
        else:
            new_points = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points)  # [B, 3+C, N, S]

        # new_xyz: sampled points position data, [B, C, npoint]
        # new_points: sampled points data, [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            if self.use_act:
                if self.bn:
                    bn = self.mlp_bns[i]
                    new_points = self.act(bn(conv(new_points)))
                else:
                    new_points = self.act(conv(new_points))
            else:
                new_points = conv(new_points)

        if self.mean_aggr:
            new_points = torch.mean(new_points, -1)
        else:
            new_points = torch.max(new_points, -1)[0]

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.use_act:
                    if self.bn:
                        bn = self.mlp2_bns[i]
                        new_points = self.act(bn(conv(new_points)))
                    else:
                        new_points = self.act(conv(new_points))
                else:
                    new_points = conv(new_points)

        if self.return_fps:
            return new_xyz, new_points, fps_idx
        else:
            return new_xyz, new_points


class Mapping(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True, return_inter=False, radius=False):
        super(Mapping,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.return_inter = return_inter
        self.mlp_convs = nn.ModuleList()
        self.mlp2 = mlp2
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel  + latent_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm1d(out_channel))

                last_channel = out_channel

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        sqrdists = square_distance(xyz1, xyz2)
        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
        
        # dists, knn_idx = torch3d.knn_points(xyz1, xyz2, self.nsample)

        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]

        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)
        inter = new_points

        if self.mlp2:
            for i, conv in enumerate(self.mlp2_convs):
                if self.bn:
                    bn = self.mlp2_bns[i]
                    new_points =  self.relu(bn(conv(new_points)))
                else:
                    new_points =  self.relu(conv(new_points))
                    # new_points =  self.relu(self.drop(conv(new_points)))
        if self.return_inter:
            return new_points, inter
        return new_points
