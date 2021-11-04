#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import itertools


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
    device = torch.device('cuda')

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


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.leaky_relu(self.w_1(x), negative_slope=0.2)))


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class DGCNN(nn.Module):
    def __init__(self, args, conv5_dim=1024):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.conv5_dim = conv5_dim
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(conv5_dim)
        self.bn6 = nn.BatchNorm1d(2048)
        self.bn7 = nn.BatchNorm1d(2048)

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
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.conv5_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(1024, 2048, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(2048, 2048, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

    def extract_feature(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # 1024*6*20
        x = self.conv1(x)  # 1024*64*20
        x1 = x.max(dim=-1, keepdim=False)[0]  # 1024*64

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

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # 8 * 1024

        return x1

    def forward(self, original_partial_pc, transformed_partial_pc, FPS1_pc, FPS2_pc, FPS3_pc, FPS1_trans_partial_PC, FPS2_trans_partial_PC, FPS3_trans_partial_PC):
        """
        extract global features
        :param x:
        :return:
        """
        #complete_x1 = self.extract_feature(complete_pc)
        original_partial_global_feature = self.extract_feature(original_partial_pc)
        original_partial_pc_x1 = torch.unsqueeze(original_partial_global_feature, 1)
        FPS1_x1 = torch.unsqueeze(self.extract_feature(FPS1_pc), 1)
        FPS2_x1 = torch.unsqueeze(self.extract_feature(FPS2_pc), 1)
        FPS3_x1 = torch.unsqueeze(self.extract_feature(FPS3_pc), 1)
        original_partial_feature = torch.cat((original_partial_pc_x1, FPS1_x1, FPS2_x1, FPS3_x1), dim=1)

        transformed_partial_global_feature = self.extract_feature(transformed_partial_pc)
        transformed_partial_pc_x1 = torch.unsqueeze(transformed_partial_global_feature, 1)
        FPS1_trans_partial_PC_x1 = torch.unsqueeze(self.extract_feature(FPS1_trans_partial_PC), 1)
        FPS2_trans_partial_PC_x1 = torch.unsqueeze(self.extract_feature(FPS2_trans_partial_PC), 1)
        FPS3_trans_partial_PC_x1 = torch.unsqueeze(self.extract_feature(FPS3_trans_partial_PC), 1)
        transformed_partial_feature = torch.cat((transformed_partial_pc_x1, FPS1_trans_partial_PC_x1, FPS2_trans_partial_PC_x1, FPS3_trans_partial_PC_x1), dim=1)

        return original_partial_pc_x1, transformed_partial_pc_x1, original_partial_feature, transformed_partial_feature


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.N = args.n_blocks
        self.dropout_T = args.dropout_T
        self.n_ff_dims = args.n_ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.n_emb_dims)
        ff = PositionwiseFeedForward(self.n_emb_dims, self.n_ff_dims, self.dropout_T)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.n_emb_dims, c(attn), c(ff), self.dropout_T), self.N),
                                    Decoder(DecoderLayer(self.n_emb_dims, c(attn), c(attn), c(ff), self.dropout_T), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]  # 768*512
        tgt = input[1]  # 768*512
        src = src.transpose(2, 1).contiguous()  # 512*768
        tgt = tgt.transpose(2, 1).contiguous()  # 512*768
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        # self.q_conv.conv.weight = self.k_conv.conv.weight
        self.v_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.trans_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c
        x_k = self.k_conv(x)# b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n
        attention = self.softmax(energy)
        x_r = torch.bmm(x_v, attention) # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class netGNet(nn.Module):
    def __init__(self, args):
        super(netGNet, self).__init__()
        self.args = args
        self.meshgrid = [[-0.3, 0.3, 32], [-0.3, 0.3, 32]]
        self.k = args.k
        self.num_coarse = args.num_coarse
        self.DGCNN = DGCNN(args)
        self.attention = Transformer(self.args)
        self.SA = SA_Layer(1024)
        self.mlp1 = nn.Sequential(
            # nn.Conv1d(2048, 1024, kernel_size=1),
            # nn.ReLU(),
            # nn.Conv1d(1024, 512, kernel_size=1),
            nn.Conv1d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 3, kernel_size=1),
        )
        self.mlp2 = nn.Sequential(
            # nn.Conv1d(2565, 1024, kernel_size=1),
            # nn.ReLU(),
            # nn.Conv1d(1024, 512, kernel_size=1),
            nn.Conv1d(1541, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 3, kernel_size=1),
        )

        self.maxpool = nn.AdaptiveMaxPool1d(512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.conv = nn.Sequential(nn.Conv1d(2048, 512, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(negative_slope=0.2))

    def build_grid(self, batch_size):
        # ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in self.meshgrid])
        # ndim = len(self.meshgrid)
        # grid = np.zeros((np.prod([it[2] for it in self.meshgrid]), ndim), dtype=np.float32)  # MxD
        # for d in range(ndim):
        #     grid[:, d] = np.reshape(ret[d], -1)
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        grid = np.array(list(itertools.product(x, y)))  # 32*32
        grid = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)  # b*45*45
        grid = torch.tensor(grid)
        return grid.float()


    def forward(self, *input):
        """
        Encoder
        """
        original_partial_global_feature, transformed_partial_global_feature, original_partial_feature, transformed_partial_feature = self.DGCNN(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7])
        original_partial_feature_p, transformed_partial_feature_p = self.attention(original_partial_feature.transpose(1, 2), transformed_partial_feature.transpose(1, 2))  # b*1024*4
        original_partial_feature_p = (original_partial_feature + original_partial_feature_p.transpose(1, 2))  # b*4*1024
        transformed_partial_feature_p = (transformed_partial_feature + transformed_partial_feature_p.transpose(1, 2))  # b*4*1024
        """
        Decoder:coarse output
        """
        original_partial_feature_repeatM = original_partial_feature_p.repeat(1, int(self.num_coarse/4), 1)  # b*m*1024
        coarse_output1 = self.mlp1(original_partial_feature_repeatM.transpose(1, 2)).transpose(1, 2)  # b*m*3

        transformed_partial_feature_repeatM = transformed_partial_feature_p.repeat(1, int(self.num_coarse/4), 1)  # b*m*1024
        coarse_output2 = self.mlp1(transformed_partial_feature_repeatM.transpose(1, 2)).transpose(1, 2)  # b*m*3
        """
        self-attention
        """
        original_partial_feature_SA = self.SA(original_partial_feature.transpose(1, 2)).transpose(1, 2)  # b*4*1024
        transformed_partial_feature_SA = self.SA(transformed_partial_feature.transpose(1, 2)).transpose(1, 2)  # b*4*1024
        """
        Decoder:fine output
        """
        coarse_output1_tile = coarse_output1.repeat(1, 2, 1)  # b*2m*3
        coarse_output2_tile = coarse_output2.repeat(1, 2, 1)  # b*2m*3

        original_partial_feature_pool = self.maxpool(original_partial_feature_p)  # b*4*512
        transformed_partial_feature_pool = self.maxpool(transformed_partial_feature_p)

        original_partial_feature_pool_tile = original_partial_feature_pool.repeat(1, int(self.num_coarse/2), 1)  # b*2m*512
        transformed_partial_feature_pool_tile = transformed_partial_feature_pool.repeat(1, int(self.num_coarse/2), 1)  # b*2m*512
        original_partial_feature_SA_tile = original_partial_feature_SA.repeat(1, int(self.num_coarse/2), 1)  # b*2m*1024
        transformed_partial_feature_SA_tile = transformed_partial_feature_SA.repeat(1, int(self.num_coarse/2), 1)
        batch_size = coarse_output1_tile.shape[0]
        grid = self.build_grid(batch_size)  # [b, 2m, 2]
        if torch.cuda.is_available():
            grid = grid.cuda()
        # print('coarse_output1_tile size: ', coarse_output1_tile.size())
        # print('original_partial_feature_pool_tile size: ', original_partial_feature_pool_tile.size())
        # print('grid size: ', grid.size())
        # print('original_partial_feature_SA_tile size: ', original_partial_feature_SA_tile.size())

        fine_feature_map1 = torch.cat((coarse_output1_tile, original_partial_feature_pool_tile, grid, original_partial_feature_SA_tile), dim=-1).transpose(1, 2)  # b*1541*2m
        fine_feature_map2 = torch.cat((coarse_output2_tile, transformed_partial_feature_pool_tile, grid, transformed_partial_feature_SA_tile), dim=-1).transpose(1, 2)  # b*1541*2m

        fine_output1 = self.mlp2(fine_feature_map1).transpose(1, 2)  # b*2m*3
        fine_output2 = self.mlp2(fine_feature_map2).transpose(1, 2)  # b*2m*3
        return coarse_output1, coarse_output2, fine_output1, fine_output2


class netDNet(nn.Module):
    def __init__(self, args):
        super(netDNet, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(256, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(64, 1)
        self.bn8 = nn.BatchNorm1d(1)

    def forward(self, x):
        # print('x:',x.size())
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

        x = self.conv5(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # b*1*1024

        # x_linear1 = self.linear1(x)
        # x = F.leaky_relu(x_linear1, negative_slope=0.2)  # b*256
        x_bn6 = self.bn6(self.linear1(x))
        x = F.leaky_relu(x_bn6, negative_slope=0.2)  # b*256
        x = self.dp1(x)

        # x_linear2 = self.linear2(x)

        x_linear2 = self.bn7(self.linear2(x))

        x = F.leaky_relu(x_linear2, negative_slope=0.2)  # b*64
        x = self.dp2(x)
        # x = F.sigmoid(self.linear3(x))  # b*1
        # x = F.relu(self.linear3(x)) # b*1
        x = self.linear3(x)  # b*1
        # print(x)
        return x


