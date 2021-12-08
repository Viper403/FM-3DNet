import os
import argparse
from torch.utils.data import DataLoader
from data import ModelNet40
import torch
import open3d as o3d
from RegModel import RegModel


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
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='registration', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='/home/huijie/research/EECS542/FM-3DNet/checkpoints/registration/models/model.best.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--pre_model_path', type=str, default='', metavar='N',
                        help='Pretrained DGCNN path')
    args = parser.parse_args()

    train_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='train', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if args.cuda else "cpu")
    net = RegModel(args).cuda()
    if args.model_path:
        if os.path.isfile(args.model_path):
            print("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            net.load_state_dict(checkpoint)
            # model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.model_path))
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