import os
import shutil
import json

import numpy as np
from tqdm import tqdm

import irsg_core.data_pull as dp
from config import get_config_path

TRAIN_SPLIT = 0.8

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']
    data_path = cfg_data['file_paths']['mat_path']
    sg_path = cfg_data['file_paths']['sg_path']


def generate_rcnn_data(ifd_train, ifd_test):
    dataset_path = os.path.join(out_path, 'IRSG_RCNN')
    anno_dir = os.path.join(dataset_path, 'Annotations')
    image_dir = os.path.join(dataset_path, 'Images')
    set_dir = os.path.join(dataset_path, 'ImageSets')

    for path in (anno_dir, image_dir, set_dir):
        if not os.path.exists(path):
            os.makedirs(path)

    full_train_size = len(ifd_train.vg_data)
    val_size = int(full_train_size * TRAIN_SPLIT)
    train_index = np.random.choice(full_train_size, size=val_size)
    val_index = np.array([i for i in np.arange(full_train_size)
                          if i not in train_index])

    if_datas = (ifd_train, ifd_train, ifd_test)
    vg_datas = (ifd_train.vg_data[train_index], ifd_train.vg_data[val_index],
                ifd_test.vg_data)
    set_names = ('train.txt', 'val.txt', 'test.txt')
    descs = ('train', 'val', 'test')
    image_index = 0
    for if_data, vg_data, set_name, desc in zip(
            if_datas, vg_datas, set_names, descs):
        index_list = []
        for _ in tqdm(range(len(vg_data)), desc=desc):
            import pdb; pdb.set_trace()
            if_data.configure(image_index, None, load_all=True)
            image_name = '{:06d}.jpg'.format(image_index)
            image_path = os.path.join(image_dir, image_name)
            shutil.copy(if_data.image_filename, image_path)
            index_list.append('{:06d}'.format(image_index))
            image_index += 1
        set_path = os.path.join(set_dir, set_name)
        with open(set_path, 'w') as f:
            for index in index_list:
                f.write('{}\n'.format(index))


if __name__ == '__main__':
    if_data_test = dp.get_ifdata(use_csv=True)
    if_data_train = dp.get_ifdata(use_csv=True, use_train=True)
    generate_rcnn_data(if_data_train, if_data_test)
