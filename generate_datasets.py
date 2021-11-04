import os
import glob
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from scipy.spatial.transform import Rotation


def load_data(partition):
    """
    读取h5文件中的data和label两个数据集到列表中
    :param partition: h5文件名
    :return: data数据列表和label数据列表
    """
    DATA_DIR = 'data'
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    """
    平移点云
    :param pointcloud: 要平移的目标点云
    :return: 平移之后的点云
    """
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

def transformPointcloud(partial_cloud, rot_factor=4, d=0.5):
    """

    :param d:
    :param partial_cloud:
    :param rot_factor:
    :return:
    """

    """
    rotation
    """
    anglex = np.random.uniform() * np.pi / rot_factor  # x轴的旋转角度
    angley = np.random.uniform() * np.pi / rot_factor  # y轴的旋转角度
    anglez = np.random.uniform() * np.pi / rot_factor  # z轴的旋转角度
    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])  # 沿x轴旋转矩阵
    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])  # 沿y轴旋转矩阵
    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])  # 沿z轴旋转矩阵
    R_ab = Rx.dot(Ry).dot(Rz) # 点云P到Q的旋转矩阵
    R_ba = R_ab.T # 点云q到P旋转矩阵

    """
    translation
    """
    translation_ab = np.array([np.random.uniform(-d, d), np.random.uniform(-d, d),
                               np.random.uniform(-d, d)])  # 沿x， y， z的平移量 由p到q
    translation_ba = -R_ba.dot(translation_ab)  # 由q到p的平移

    partial_cloud = partial_cloud.T
    rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
    pointcloud_ = rotation_ab.apply(partial_cloud.T).T + np.expand_dims(translation_ab, axis=1)  # 转换后的点云

    euler_ab = np.asarray([anglez, angley, anglex])  # 三个轴旋转角度
    euler_ba = -euler_ab[::-1]
    return pointcloud_, translation_ba, euler_ba


def pairing(start_index, end_index):
    data, label = load_data('train')
    # transformed_partial_clouds = list()
    translation_list = list()
    rotation_list = list()
    point_clouds_list = list()
    transformed_point_clouds_list = list()
    for cloud_index in range(start_index, end_index):  # data.shape[0]
        pointcloud = data[cloud_index][:1024]
        pointcloud = translate_pointcloud(pointcloud)
        point_clouds_list.append(pointcloud)
        transformed_pointcloud, translation_ba, euler_ba = transformPointcloud(pointcloud)
        transformed_pointcloud = transformed_pointcloud.T
        transformed_point_clouds_list.append(transformed_pointcloud)
        translation_list.append(translation_ba)
        rotation_list.append(euler_ba)
    return point_clouds_list, transformed_point_clouds_list, translation_list, rotation_list#


def saveH5(point_clouds_list, transformed_point_clouds_list, translation_list, rotation_list, file_path):
    hdfFile = h5py.File(file_path, 'w')
    translation_array = np.array(translation_list)
    rotation_array = np.array(rotation_list)

    hdfFile.create_dataset('point_clouds', data=np.array(point_clouds_list))
    hdfFile.create_dataset('transformed_point_clouds', data=np.array(transformed_point_clouds_list))
    hdfFile.create_dataset('translation', data=translation_array)
    hdfFile.create_dataset('rotation', data=rotation_array)
    hdfFile.close()


def readH5():
    h5_name = "./data.h5"
    f = h5py.File(h5_name)
    partial_data = f.get("partialPointcloud_0")
    complete_data = f.get("completePointcloud")
    f.close()


def data_preprocess(partition):
    # start_index_list = [0, 6, 11, 17, 23]
    # end_index_list = [5, 10, 16, 22, 28]
    if partition=='train':
        start_index_list = [0, 2048, 4096, 6144, 8192]
        end_index_list = [2048, 4096, 6144, 8192, 9840]
        for h5_index in range(0, 5):
            point_clouds_list, transformed_point_clouds_list, translation_list, rotation_list = pairing(start_index_list[h5_index], end_index_list[h5_index])
            # save H5files
            saveH5(point_clouds_list, transformed_point_clouds_list, translation_list, rotation_list, "trainData_{}.h5".format(str(h5_index)))
    else:
        point_clouds_list, transformed_point_clouds_list, translation_list, rotation_list = pairing(0,2048)
        saveH5(point_clouds_list, transformed_point_clouds_list, translation_list, rotation_list,"testData.h5")
if __name__ == '__main__':
    data_preprocess('test')
    #readH5()
