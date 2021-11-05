import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


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


def load_data(partition, debug):
    # download()
    if debug:
        DATA_DIR = './data/debug'
    else:
        DATA_DIR = './data'
    all_point_clouds = []
    all_transformed_point_clouds = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, '%sData_*.h5'%partition)):
        f = h5py.File(h5_name)
        point_clouds = f['point_clouds'][:].astype('float32')
        transformed_point_clouds = f['transformed_point_clouds'][:].astype('float32')
        f.close()
        all_point_clouds.append(point_clouds)
        all_transformed_point_clouds.append(transformed_point_clouds)
    all_point_clouds = np.concatenate(all_point_clouds, axis=0)
    all_transformed_point_clouds = np.concatenate(all_transformed_point_clouds, axis=0)
    return all_point_clouds, all_transformed_point_clouds


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', debug = False):
        self.point_clouds, self.transformed_point_clouds = load_data(partition, debug)
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.point_clouds[item]
        transformed_point_cloud = self.transformed_point_clouds[item]
        if self.partition == 'train':
            #pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
            np.random.shuffle(transformed_point_cloud)
        return pointcloud, transformed_point_cloud

    def __len__(self):
        return self.point_clouds.shape[0]


if __name__ == '__main__':
    train = ModelNet40('train')
    #test = ModelNet40()
    for pc1, pc2 in train:
        print(pc1)
        print(pc2)
        break
