import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import pytorch_lightning as pl

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    # ximin
    device = torch.device('cuda')
    # device = torch.device("cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        #self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        #self.bn6 = nn.BatchNorm1d(512)
        #self.dp1 = nn.Dropout(p=args.dropout)
        #self.linear2 = nn.Linear(512, 256)
        #self.bn7 = nn.BatchNorm1d(256)
        #self.dp2 = nn.Dropout(p=args.dropout)
        #self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)   #batch*1024*1024
        '''
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        '''
        return x


class norm(nn.Module):
    def __init__(self, axis=2):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        mean = torch.mean(x, self.axis,keepdim=True)
        std = torch.std(x, self.axis,keepdim=True)
        x = (x-mean)/(std+1e-6)
        return x


class Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input*8
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Modified_softmax(nn.Module):
    def __init__(self, axis=1):
        super(Modified_softmax, self).__init__()
        self.axis = axis
        self.norm = norm(axis = axis)
    def forward(self, x):
        x = self.norm(x)
        x = Gradient.apply(x)
        x = F.softmax(x, dim=self.axis)
        return x


class FM3D(nn.Module):
    def __init__(self, args):
        super(FM3D, self).__init__()
        self.args = args
        self.DGCNN = DGCNN(args)
        self.DeSmooth = nn.Sequential(
            #nn.Conv1d(in_channels=self.input_pts, out_channels=self.input_pts + 128, kernel_size=1, stride=1,
            #          bias=False),
            #nn.ReLU(),
            norm(axis=1),
            #nn.Conv1d(in_channels=self.input_pts + 128, out_channels=self.input_pts, kernel_size=1, stride=1,
            #          bias=False),
            Modified_softmax(axis=2)
        )
        self.bn1 = nn.BatchNorm1d(args.emb_dims//2)
        self.bn2 = nn.BatchNorm1d(args.emb_dims)
        self.predictor = nn.Sequential(nn.Conv1d(args.emb_dims, args.emb_dims//2, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Conv1d(args.emb_dims//2, args.emb_dims, kernel_size=1, bias=False),
                                        self.bn2,
                                        nn.LeakyReLU(negative_slope=0.2))
    def _KFNN(self, x, y, k=10):
        def batched_pairwise_dist(a, b):
            x, y = a.float(), b.float()
            bs, num_points_x, points_dim = x.size()
            bs, num_points_y, points_dim = y.size()
            assert(num_points_x==1024 or num_points_x==256)

            xx = torch.pow(x, 2).sum(2)
            yy = torch.pow(y, 2).sum(2)
            zz = torch.bmm(x, y.transpose(2, 1))   #bmm: multiple (b,w,h) with (b,h,w)
            rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x)
            ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y)
            P = rx.transpose(2, 1) + ry - 2 * zz
            return P

        pairwise_distance = batched_pairwise_dist(x.permute(0,2,1), y.permute(0,2,1))
        similarity=-pairwise_distance
        idx = similarity.topk(k=k, dim=-1)[1]
        return pairwise_distance, idx

    def forward(self,pointcloud,transformed_pointcloud):
        fe1 = self.DGCNN(pointcloud)
        fe2 = self.DGCNN(transformed_pointcloud)   #b*d*n
        pairwise_distance,_ = self._KFNN(fe1,fe2)
        similarity = 1 / (pairwise_distance + 1e-6) #b*n*n
        M = self.DeSmooth(similarity.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  #b*n*n
        M_t = M.transpose(2, 1).contiguous()  #which one is which one?
        fe1_permuted = torch.bmm(fe1, M)
        fe2_permuted = torch.bmm(fe2, M_t)  #batch*num_points*feature_dimension

        fe1_final = self.predictor(fe1_permuted)
        fe2_final = self.predictor(fe2_permuted) #b*d*n

        fe1_nograd = fe1.detach()
        fe2_nograd = fe2.detach()

        return fe1_nograd, fe2_nograd, fe1_final, fe2_final, M


class contrastive_loss(nn.Module):
    def __init__(self, args):
        super(contrastive_loss, self).__init__()
        self.args = args
        self.alpha1 = args.alpha1
        self.alpha2 = args.alpha2

    def _batch_frobenius_norm(self, matrix1, matrix2):
        loss_F = torch.norm((matrix1-matrix2),dim=(1,2))
        return loss_F

    def forward(self, fe1_nograd, fe2_nograd, fe1_final, fe2_final, M):
        batch_size = fe1_final.shape[0]
        bmean_loss_F1 = torch.mean(self._batch_frobenius_norm(fe1_nograd, fe2_final))
        bmean_loss_F2 = torch.mean(self._batch_frobenius_norm(fe2_nograd, fe1_final))
        # ximin
        I_N1 = torch.eye(n=M.shape[2]).cuda()
        # I_N1 = torch.eye(n=M.shape[2])
        
        I_N1 = I_N1.unsqueeze(0).repeat(batch_size, 1, 1)
        M_loss1 = torch.mean(
            self._batch_frobenius_norm(torch.bmm(M, M.transpose(2, 1).contiguous()), I_N1.float()))
        M_loss2 = torch.mean(torch.norm(M,dim=(1,2)))
        FB_loss = (bmean_loss_F1+bmean_loss_F2)/2
        final_loss = FB_loss + self.alpha1*M_loss1 +self.alpha2*M_loss2

        return final_loss, FB_loss, M_loss1, M_loss2




