import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import cv2
import json


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


def load_data(partition, debug,path):
    # download()
    if debug:
        DATA_DIR = os.path.join(path,'debug')
    else:
        DATA_DIR = path
    all_point_clouds = []
    all_transformed_point_clouds = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, '%sDataPart_*.h5'%partition)):
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
    def __init__(self, num_points, path, partition='train', debug = False):
        self.point_clouds, self.transformed_point_clouds = load_data(partition, debug, path)
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


def load_data_partseg(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data_part_seg')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'train*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, '%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg

def load_color_partseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/partseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    partseg_colors = np.array(colors)
    partseg_colors = partseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1350
    img = np.zeros((1350, 1890, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (1900, 1900), [255, 255, 255], thickness=-1)
    column_numbers = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    column_gaps = [320, 320, 300, 300, 285, 285]
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for row in range(0, img_size):
        column_index = 32
        for column in range(0, img_size):
            color = partseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.76, (0, 0, 0), 2)
            column_index = column_index + column_gaps[column]
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 50:
                cv2.imwrite("prepare_data/meta/partseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column + 1 >= column_numbers[row]):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice
        self.partseg_colors = load_color_partseg()
        
        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0
            
      
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]



if __name__ == '__main__':
    train = ModelNet40('train')
    #test = ModelNet40()
    for pc1, pc2 in train:
        print(pc1)
        print(pc2)
        break
