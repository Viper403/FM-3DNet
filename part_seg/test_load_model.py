import sys
sys.path.append("/home/huijie/research/EECS542/FM-3DNet")
import argparse
import torch
from model import DGCNN_partseg
import numpy as np
import os
import argparse



parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--exp_name', type=str, default='part_seg_backbone', metavar='N',
                    help='Name of the experiment')
#parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
#                    choices=['pointnet', 'dgcnn'],
#                    help='Model to use, [pointnet, dgcnn]')
parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                    choices=['modelnet40'])
parser.add_argument('--datapath',type=str, default='./data_part_seg',metavar='path')
parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=12, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--use_sgd', type=bool, default=False,
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval', type=bool,  default=False,
                    help='evaluate the model')
parser.add_argument('--num_points', type=int, default=2048,
                    help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--input_pts', type=int, default=2048, metavar='N',
                    help='#')          
parser.add_argument('--alpha1', type=float, default=0.1, metavar='N',
                    help='#')   
parser.add_argument('--alpha2', type=float, default=0.1, metavar='N',
                    help='#')         
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--debug', type=bool, default=False,
                    help='Debug mode')
args = parser.parse_args()
path = "./part_seg/models/40.pth"
checkpoint = torch.load(path)

model = DGCNN_partseg(args)

print(checkpoint.keys())
model.load_state_dict(checkpoint['DGCNN_state_dict'])
