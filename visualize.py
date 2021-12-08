import os
import argparse
from torch.utils.data import DataLoader
from data import ModelNet40WithSequence
from model import FM3D
import torch
import open3d as o3d

def visualize(pc, trans_pc, M):
    colors = (((pc.T - pc.min())/(pc.max() - pc.min()))*0.5 + 0.5).to(torch.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.T)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    trans_pcd = o3d.geometry.PointCloud()
    trans_pcd.points = o3d.utility.Vector3dVector(trans_pc.T)
    trans_pcd.colors = o3d.utility.Vector3dVector(colors)
    prob, correspoinding = torch.max(M, axis = 1)
    points = torch.concat((pc.T, trans_pc.T)).tolist()
    lines = []
    line_num = 0
    line_num_max = 100
    colors_lines = []
    for index_main, index_trans in enumerate(correspoinding):
        if prob[index_main] > 0.4 and line_num < line_num_max:
            line_num += 1
            lines.append([index_main, index_trans.item() + pc.shape[1]])
            colors_lines.append(colors[index_trans].tolist())
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors_lines)
    o3d.visualization.draw_geometries([pcd, trans_pcd, line_set])
        

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp_use_exponential', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--input_pts', type=int, default=1024, metavar='N',
                        help='#')          
    parser.add_argument('--alpha1', type=float, default=0.1, metavar='N',
                        help='#')   
    parser.add_argument('--alpha2', type=float, default=0.1, metavar='N',
                        help='#')         
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='/home/huijie/research/EECS542/FM-3DNet/checkpoints/exp/models/50.pth', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Debug mode')
    parser.add_argument('--similarity_metric', type=str, default='exponential', metavar='N',
                        help='how to measure similarity: exponential or reciprocal')
    args = parser.parse_args()

    train_loader = DataLoader(ModelNet40WithSequence(partition='train', num_points=args.num_points, debug = args.debug), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = FM3D(args).to(device)
    if args.model_path:
        if os.path.isfile(args.model_path):
            print("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            args.start_epoch = checkpoint['epoch']
            model.DGCNN.load_state_dict(checkpoint['DGCNN_state_dict'])
            model.predictor.load_state_dict(checkpoint['predictor_state_dict'])
            # model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))
    with torch.no_grad():
        for pointcloud, transformed_point_cloud, index in train_loader:
            pointcloud = pointcloud.to(device)  #b*1024*3
            transformed_point_cloud = transformed_point_cloud.to(device)
            pointcloud = pointcloud.permute(0, 2, 1) #b*3*1024
            transformed_point_cloud = transformed_point_cloud.permute(0, 2, 1)
            batch_size = pointcloud.size()[0]  
            _, _, _, _, M = model(pointcloud, transformed_point_cloud)
            M_host = torch.squeeze(M.to("cpu"))
            pointcloud_host = torch.squeeze(pointcloud.to("cpu"))
            transformed_point_cloud_host = torch.squeeze(transformed_point_cloud.to("cpu"))
            visualize(pointcloud_host, transformed_point_cloud_host - 1.1, M_host)
            pass