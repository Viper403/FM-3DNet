from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3 "
import argparse
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
import numpy as np
from torch.utils.data import DataLoader
from model import netGNet, netDNet
from util import PointLoss
import sklearn.metrics as metrics
import random
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import data_parallel_my_v2
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def plot_Loss(title, loss_list, save_path):
    plt.xticks(rotation=45)
    plt.xlabel("epoch")
    plt.ylabel("loss value")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.plot(loss_list)
    plt.savefig(save_path)
    plt.close()

def saveH5(fine_output, file_path):
    """
    save fine output
    :param fine_output: b*m*3的tensor
    :param file_path:
    :return:
    """
    hdfFile = h5py.File(file_path, 'w')
    fine_output = np.array(fine_output.cpu().detach().numpy())
    hdfFile.create_dataset('fine_output', data=fine_output)

    hdfFile.close()


def train(args):
    train_loader = DataLoader(ModelNet40(partition='train'), num_workers=4,
                              batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True) #6 data 2 label
    test_loader = DataLoader(ModelNet40(partition='test'), num_workers=4,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=True, pin_memory=True)
    print("len train_loader: ", len(train_loader))
    print("len test_loader: ", len(test_loader))
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    gpu0_bsz = 2
    acc_grad = 8
    '''
    正常的parallel版本
    # point_netG = netGNet(args).to(device)
    # point_netD = netDNet(args).to(device)
    # point_netG = torch.nn.DataParallel(point_netG)
    # point_netD = torch.nn.DataParallel(point_netD)
    '''
    point_netG = netGNet(args)
    point_netD = netDNet(args)
    point_netG = data_parallel_my_v2.BalancedDataParallel(gpu0_bsz // acc_grad, point_netG, dim=0).cuda()
    point_netD = data_parallel_my_v2.BalancedDataParallel(gpu0_bsz // acc_grad, point_netD, dim=0).cuda()
    # torch.distributed.init_process_group(backend="nccl")
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # point_netG = torch.nn.parallel.DistributedDataParallel(point_netG)
    # point_netD = torch.nn.parallel.DistributedDataParallel(point_netD)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
    # two parts of loss functions
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_PointLoss = PointLoss().to(device)

    # setup optimizer
    optimizerD = torch.optim.Adam(point_netD.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-05,
                                  weight_decay=args.weight_decay)
    optimizerG = torch.optim.Adam(point_netG.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-05,
                                  weight_decay=args.weight_decay)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

    real_label = 0.95
    fake_label = 0

    label = torch.FloatTensor(args.batch_size)

    ###########################
    #  G-NET and T-NET
    ##########################
    if args.D_choose == 1:
        Err_D_list, Err_G_D_list, Err_G_12_list, Err_G_list, train_fine_loss_list = [], [], [], [], []
        err_D_ave, err_G_D_ave, errG_12_ave, err_G_ave, train_fine_loss_ave = 0, 0, 0, 0, 0
        test_fine_loss_list = []
        test_fine_loss_ave = 0
        save_flag = 0
        for epoch in range(args.epochs):
            f1 = open('./loss/loss_G.txt', 'a')
            f2 = open('./loss/loss_D.txt', 'a')
            f3 = open('./loss/loss_G_D.txt', 'a')
            f4 = open('./loss/loss_G_12.txt', 'a')
            f5 = open('./loss/loss_train_fine.txt', 'a')
            f6 = open('./loss/loss_test_fine.txt', 'a')
            if epoch < 30:
                alpha1 = 0.01  # might need future adjustment
            elif epoch < 80:
                alpha1 = 0.05
            else:
                alpha1 = 0.1

            datagroup_index = 0
            for complete_Pointcloud, transformed_complete_PC, original_partial_PC, transformed_partial_PC, FPS1_partial_PC, FPS2_partial_PC, \
                FPS3_partial_PC, FPS1_trans_partial_PC, FPS2_trans_partial_PC, FPS3_trans_partial_PC, translation, rotation in train_loader:
                datagroup_index += 1
                complete_Pointcloud = complete_Pointcloud.permute(0, 2, 1).to(device, non_blocking=True)
                transformed_complete_PC = transformed_complete_PC.permute(0, 2, 1).to(device, non_blocking=True)
                original_partial_PC = original_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                transformed_partial_PC = transformed_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS1_partial_PC = FPS1_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS2_partial_PC = FPS2_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS3_partial_PC = FPS3_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS1_trans_partial_PC = FPS1_trans_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS2_trans_partial_PC = FPS2_trans_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS3_trans_partial_PC = FPS3_trans_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                # FPS4_partial_PC = FPS4_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                # real_point, target = data
                batch_size = complete_Pointcloud.size()[0]

                '''
                generate coarse and fine outputs
                '''
                netG_input = [original_partial_PC, transformed_partial_PC, FPS1_partial_PC, FPS2_partial_PC, FPS3_partial_PC, FPS1_trans_partial_PC, FPS2_trans_partial_PC, FPS3_trans_partial_PC]
                coarse_output1, coarse_output2, fine_output1, fine_output2 = point_netG(*netG_input)


                # real_center = torch.FloatTensor(batch_size, 1, args.crop_point_num, 3) # empty vessel
                # input_cropped1 = torch.FloatTensor(batch_size, args.pnum, 3) # empty vessel
                # input_cropped1 = input_cropped1.data.copy_(real_point)
                # real_point = torch.unsqueeze(real_point, 1)
                # input_cropped1 = torch.unsqueeze(input_cropped1, 1)
                # p_origin = [0, 0, 0]
                # if opt.cropmethod == 'random_center':
                #     # Set viewpoints
                #     choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                #               torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
                #     for m in range(batch_size):
                #         index = random.sample(choice, 1)  # Random choose one of the viewpoint
                #         distance_list = []
                #         p_center = index[0]
                #         for n in range(opt.pnum):
                #             distance_list.append(distance_squre(real_point[m, 0, n], p_center))
                #         distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
                #
                #         for sp in range(opt.crop_point_num):
                #             input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
                #             real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]
                label.resize_([batch_size, 1]).fill_(real_label)
                label = label.to(device, non_blocking=True)

                point_netG = point_netG.train()
                point_netD = point_netD.train()
                ############################
                # (2) Update D network
                ###########################
                point_netD.zero_grad()
                output = point_netD(complete_Pointcloud)
                # print("complete point cloud output: ", output)
                # print("complete label: ", label)
                errD_real1 = criterion(output, label)
                errD_real1.backward()

                label.data.fill_(fake_label)
                # Here the dim of fine_output is bs*1024*3, so permute is necessary
                output = point_netD(fine_output1.permute(0, 2, 1).detach())
                # print("fine PC output1: ", output)
                # print("fine PC label: ", label)
                errD_fake1 = criterion(output, label)
                errD_fake1.backward()

                label.data.fill_(real_label)
                output = point_netD(transformed_complete_PC)
                # print("transformed_complete_PC output: ", output)
                # print("transformed_complete_PC label: ", label)
                errD_real2 = criterion(output, label)
                errD_real2.backward()

                label.data.fill_(fake_label)
                output = point_netD(fine_output2.permute(0, 2, 1).detach())
                # print("fine PC output2: ", output)
                # print("fine PC label2: ", label)
                errD_fake2 = criterion(output, label)
                errD_fake2.backward()

                errD = max(errD_real1, errD_real2) + max(errD_fake1, errD_fake2)
                optimizerD.step()
                ############################
                # (3) Update G network: maximize log(D(G(z)))
                ###########################
                point_netG.zero_grad()
                label.data.fill_(real_label)
                output = point_netD(fine_output1.permute(0, 2, 1))
                errG_D_original = criterion(output, label)
                fine_LOSS_original = criterion_PointLoss(fine_output1, complete_Pointcloud.permute(0, 2, 1))
                errG_l2_original = criterion_PointLoss(fine_output1, complete_Pointcloud.permute(0, 2, 1)) \
                          + alpha1 * criterion_PointLoss(coarse_output1, complete_Pointcloud.permute(0, 2, 1)[:][:args.num_coarse])

                errG_original = (1 - args.wtl2) * errG_D_original + args.wtl2 * errG_l2_original
                errG_original.backward(retain_graph=True)

                label.data.fill_(real_label)
                output = point_netD(fine_output2.permute(0, 2, 1))
                errG_D_transformed = criterion(output, label)
                fine_LOSS_transformed = criterion_PointLoss(fine_output2, transformed_complete_PC.permute(0, 2, 1))
                errG_l2_transformed = criterion_PointLoss(fine_output2, transformed_complete_PC.permute(0, 2, 1)) \
                          + alpha1 * criterion_PointLoss(coarse_output2, transformed_complete_PC.permute(0, 2, 1)[:][:args.num_coarse])

                errG_transformed = (1 - args.wtl2) * errG_D_transformed + args.wtl2 * errG_l2_transformed
                errG_transformed.backward()

                optimizerG.step()

                errG_D = max(errG_D_original, errG_D_transformed)
                errG_l2 = max(errG_l2_original, errG_l2_transformed)
                errG = max(errG_original, errG_transformed)
                fine_LOSS = max(fine_LOSS_original, fine_LOSS_transformed)

                # 将下面日志改成 我们的日志+可视化
                print('[%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f/ %.4f'
                      % (epoch, datagroup_index, len(train_loader),
                         errD.data, errG_D.data, errG_l2, errG, fine_LOSS))
                f = open('./loss/loss_VPGnet.txt', 'a')
                f.write('\n' + '[%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f /%.4f'
                        % (epoch,  datagroup_index, len(train_loader),
                           errD.data, errG_D.data, errG_l2, errG, fine_LOSS))
                f.close()
                if save_flag == 0:
                    saveH5(fine_output1, "./fine_output/fine_output{}.h5".format(epoch))
                    save_flag = 1
                err_D_ave += float(errD.cpu().detach().numpy())
                err_G_D_ave += float(errG_D.cpu().detach().numpy())
                errG_12_ave += float(errG_l2.cpu().detach().numpy())
                err_G_ave += float(errG.cpu().detach().numpy())
                train_fine_loss_ave += float(fine_LOSS.cpu().detach().numpy())

                # evaluation
                # test_group_loss_list = []
                # test_group_loss_ave = 0
        #if datagroup_index % 10 == 0:
            # print('After ', datagroup_index, '-th data in trainloader')
            # f.write('\n' + 'After, ' + str(datagroup_index) + '-th data in trainloader')
            print('After ', epoch, '-th data in trainloader')
            f = open('./loss/loss_evaluate.txt', 'a')
            f.write('\n' + 'After, ' + str(epoch) + '-th data in trainloader')
            test_index = 0
            # del complete_Pointcloud
            # del transformed_complete_PC
            # del original_partial_PC
            # del transformed_partial_PC
            # del FPS1_partial_PC
            # del FPS2_partial_PC
            # del FPS3_partial_PC
            # del FPS1_trans_partial_PC
            # del FPS2_trans_partial_PC
            # del FPS3_trans_partial_PC
            # del translation
            # del rotation
            for complete_Pointcloud, transformed_complete_PC, original_partial_PC, transformed_partial_PC, FPS1_partial_PC, FPS2_partial_PC, \
                FPS3_partial_PC, FPS1_trans_partial_PC, FPS2_trans_partial_PC, FPS3_trans_partial_PC, translation, rotation in test_loader:
                test_index += 1
                complete_Pointcloud = complete_Pointcloud.permute(0, 2, 1).to(device, non_blocking=True)
                transformed_complete_PC = transformed_complete_PC.permute(0, 2, 1).to(device, non_blocking=True)
                original_partial_PC = original_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                transformed_partial_PC = transformed_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS1_partial_PC = FPS1_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS2_partial_PC = FPS2_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS3_partial_PC = FPS3_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS1_trans_partial_PC = FPS1_trans_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS2_trans_partial_PC = FPS2_trans_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                FPS3_trans_partial_PC = FPS3_trans_partial_PC.permute(0, 2, 1).to(device, non_blocking=True)
                batch_size = complete_Pointcloud.size()[0]

                netG_input = [original_partial_PC, transformed_partial_PC, FPS1_partial_PC, FPS2_partial_PC,
                              FPS3_partial_PC, FPS1_trans_partial_PC, FPS2_trans_partial_PC,
                              FPS3_trans_partial_PC]
                coarse_output1, coarse_output2, fine_output1, fine_output2 = point_netG(*netG_input)
                point_netG.eval()
                #fake_center1, fake_center2, fake = point_netG(input_cropped)
                fine_test_loss1 = criterion_PointLoss(fine_output1, complete_Pointcloud.permute(0, 2, 1))
                fine_test_loss2 = criterion_PointLoss(fine_output2, transformed_complete_PC.permute(0, 2, 1))
                fine_test_loss = float(max(fine_test_loss1, fine_test_loss2).cpu().detach().numpy())
                # test_group_loss_ave += fine_test_loss
                print('test result: ', fine_test_loss)
                f.write('\n' + 'test result:  %.4f' % (fine_test_loss))
                f.close()
                #if test_index == 5:
                break
                # test_group_loss_ave /= len(test_loader)
                # test_group_loss_list.append(test_group_loss_ave)
            # test_fine_loss_ave += np.mean(test_group_loss_list)


            err_D_ave /= len(train_loader)
            err_G_D_ave /= len(train_loader)
            errG_12_ave /= len(train_loader)
            err_G_ave /= len(train_loader)
            train_fine_loss_ave /= len(train_loader)
            #test_fine_loss_ave /= len(train_loader)
            f1.write('\n' + "After " + str(epoch) + " epoch,loss G: " + str(err_G_ave))
            f2.write('\n' + "After " + str(epoch) + " epoch,loss D: " + str(err_D_ave))
            f3.write('\n' + "After " + str(epoch) + " epoch,loss G_D: " + str(err_G_D_ave))
            f4.write('\n' + "After " + str(epoch) + " epoch,loss G_12: " + str(errG_12_ave))
            f5.write('\n' + "After " + str(epoch) + " epoch,loss train fine: " + str(train_fine_loss_ave))
            #f6.write('\n' + "After " + str(epoch) + " epoch,loss test fine: " + str(test_fine_loss_ave))
            #test_fine_loss_list.append(test_fine_loss_ave)
            Err_D_list.append(err_D_ave)
            Err_G_D_list.append(err_G_D_ave)
            Err_G_12_list.append(errG_12_ave)
            Err_G_list.append(err_G_ave)
            train_fine_loss_list.append(train_fine_loss_ave)
            plot_Loss("Error_D LOSS", Err_D_list, "./fig/Error_D.jpg")
            plot_Loss("Error_G_D LOSS", Err_G_D_list, "./fig/Error_G_D.jpg")
            plot_Loss("Error_G_12 LOSS", Err_G_12_list, "./fig/Error_G_12.jpg")
            plot_Loss("Error_G LOSS", Err_G_list, "./fig/Error_G.jpg")
            plot_Loss("train_fine LOSS", train_fine_loss_list, "./fig/train_fine.jpg")
            #plot_Loss("test_fine LOSS", test_fine_loss_list, "./fig/test_fine.jpg")
            schedulerD.step()
            schedulerG.step()

            if epoch % 10 == 0:
                torch.save({'epoch': epoch + 1,
                            'state_dict': point_netG.state_dict()},
                           './Trained_Model/point_netG' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': point_netD.state_dict()},
                           './Trained_Model/point_netD' + str(epoch) + '.pth')
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            f5.close()
            #f6.close()
            print("epoch {} finished !".format(epoch))
            time.sleep(5)




if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--num_coarse', type=int, default=512, metavar='N',
                        help='points number of the coarse output ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')  # seed from DGCNN
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--n_emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--dropout_T', type=float, default=0.0,
                        help='dropout rate of Transformer')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--n_ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    # parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
    #                     help='Dimension of embeddings')
    parser.add_argument('--D_choose', type=int, default=1, help='0 not use D-net,1 use D-net')
    parser.add_argument('--wtl2', type=float, default=0.95, help='0 means do not use else use with this weight')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    # parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    _init_()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    train(args)