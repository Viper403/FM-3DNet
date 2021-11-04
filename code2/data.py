#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    # download()
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = "./data/data_all"
    complete_datas = []
    transformed_complete_PCs = []
    original_partial_PCs = []
    Transformed_Partial_PCs = []
    FPS1_partial_PCs = []
    FPS2_partial_PCs = []
    FPS3_partial_PCs = []
    TransformedSampled_PCs_0 = []
    TransformedSampled_PCs_1 = []
    TransformedSampled_PCs_2 = []
    translations = []
    rotations = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        complete_data = f['complete_Pointcloud'][:].astype('float32')  # n*1024*3
        transformed_complete_PC = f['transformed_complete_Pointcloud'][:].astype('float32')  # n*1024*3
        original_partial_PC = f["Original_Partial_Pointcloud"][:].astype("float32")  # n*768*3
        Transformed_Partial_PC = f["Transformed_Partial_Pointcloud"][:].astype("float32")  # n*768*3
        FPS1_partial_PC = f["SampledPointcloud_0"][:].astype("float32")  # n*600*3
        FPS2_partial_PC = f["SampledPointcloud_1"][:].astype("float32")  # n*500*3
        FPS3_partial_PC = f["SampledPointcloud_2"][:].astype("float32")  # n*400*3
        TransformedSampled_PC_0 = f["TransformedSampledPointcloud_0"][:].astype("float32")  # n*600*3
        TransformedSampled_PC_1 = f["TransformedSampledPointcloud_1"][:].astype("float32")  # n*600*3
        TransformedSampled_PC_2 = f["TransformedSampledPointcloud_2"][:].astype("float32")  # n*600*3
        translation = f["translation"][:].astype("float32")  # n*1*3
        rotation = f["rotation"][:].astype("float32")  # n*1*3
        f.close()

        complete_datas.append(complete_data)  # (h5_number*n)*1024*3
        transformed_complete_PCs.append(transformed_complete_PC)
        original_partial_PCs.append(original_partial_PC)
        Transformed_Partial_PCs.append(Transformed_Partial_PC)
        FPS1_partial_PCs.append(FPS1_partial_PC)
        FPS2_partial_PCs.append(FPS2_partial_PC)
        FPS3_partial_PCs.append(FPS3_partial_PC)
        TransformedSampled_PCs_0.append(TransformedSampled_PC_0)
        TransformedSampled_PCs_1.append(TransformedSampled_PC_1)
        TransformedSampled_PCs_2.append(TransformedSampled_PC_2)
        translations.append(translation)
        rotations.append(rotation)

    complete_datas = np.concatenate(complete_datas, axis=0)
    transformed_complete_PCs = np.concatenate(transformed_complete_PCs, axis=0)
    original_partial_PCs = np.concatenate(original_partial_PCs, axis=0)
    Transformed_Partial_PCs = np.concatenate(Transformed_Partial_PCs, axis=0)
    FPS1_partial_PCs = np.concatenate(FPS1_partial_PCs, axis=0)
    FPS2_partial_PCs = np.concatenate(FPS2_partial_PCs, axis=0)
    FPS3_partial_PCs = np.concatenate(FPS3_partial_PCs, axis=0)
    TransformedSampled_PCs_0 = np.concatenate(TransformedSampled_PCs_0, axis=0)
    TransformedSampled_PCs_1 = np.concatenate(TransformedSampled_PCs_1, axis=0)
    TransformedSampled_PCs_2 = np.concatenate(TransformedSampled_PCs_2, axis=0)

    translations = np.concatenate(translations, axis=0)
    rotations = np.concatenate(rotations, axis=0)
    return complete_datas, transformed_complete_PCs, original_partial_PCs, Transformed_Partial_PCs, FPS1_partial_PCs, FPS2_partial_PCs, FPS3_partial_PCs, TransformedSampled_PCs_0, TransformedSampled_PCs_1, TransformedSampled_PCs_2, translations, rotations


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, partition='train'):
        self.complete_datas, self.transformed_complete_PCs, self.original_partial_PCs, self.Transformed_Partial_PCs, self.FPS1_partial_PCs, self.FPS2_partial_PCs, self.FPS3_partial_PCs, self.TransformedSampled_PCs_0, self.TransformedSampled_PCs_1, self.TransformedSampled_PCs_2, self.translations, self.rotations = load_data(partition)
        self.partition = partition

    def __getitem__(self, item):
        complete_data = self.complete_datas[item]  # 1*1024*3
        transformed_complete_PC = self.transformed_complete_PCs[item]
        original_partial_PC = self.original_partial_PCs[item]
        Transformed_Partial_PC = self.Transformed_Partial_PCs[item]
        FPS1_partial_PC = self.FPS1_partial_PCs[item]
        FPS2_partial_PC = self.FPS2_partial_PCs[item]
        FPS3_partial_PC = self.FPS3_partial_PCs[item]
        TransformedSampled_PC_0 = self.TransformedSampled_PCs_0[item]
        TransformedSampled_PC_1 = self.TransformedSampled_PCs_1[item]
        TransformedSampled_PC_2 = self.TransformedSampled_PCs_2[item]
        translation = self.translations[item]
        rotation = self.rotations[item]
        if self.partition == 'train':
            np.random.shuffle(complete_data)
            np.random.shuffle(transformed_complete_PC)
            np.random.shuffle(original_partial_PC)
            np.random.shuffle(Transformed_Partial_PC)
            np.random.shuffle(FPS1_partial_PC)
            np.random.shuffle(FPS2_partial_PC)
            np.random.shuffle(FPS3_partial_PC)
            np.random.shuffle(TransformedSampled_PC_0)
            np.random.shuffle(TransformedSampled_PC_1)
            np.random.shuffle(TransformedSampled_PC_2)
        return complete_data, transformed_complete_PC, original_partial_PC, Transformed_Partial_PC, FPS1_partial_PC, FPS2_partial_PC, FPS3_partial_PC, TransformedSampled_PC_0, TransformedSampled_PC_1, TransformedSampled_PC_2, translation, rotation

    def __len__(self):
        return self.complete_datas.shape[0]

if __name__ == '__main__':
    pass
    # train = ModelNet40(1024)
    # train_loader = DataLoader(ModelNet40(partition='train'), num_workers=8,
    #                           batch_size=1, shuffle=True, drop_last=True)
    # print(len(train_loader))
    # print("%%%%%%%%%%%%%%%%%%")
    # print(train_loader.shape)
    # test = ModelNet40(1024, 'test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)
