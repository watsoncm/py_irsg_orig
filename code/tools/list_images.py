import os
import json
import shutil

from tqdm import tqdm

import irsg_core.data_pull as dp
from config import get_config_path

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    sg_path = cfg_data['file_paths']['sg_path']


def copy_images_indexed(path, if_data):
    from_path = os.path.join(path, 'sg_test_images')
    to_path = os.path.join(path, 'by_index')
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    for index, image_data in tqdm(enumerate(if_data.vg_data),
                                  total=len(if_data.vg_data)):
        image_name = os.path.basename(image_data.image_path)
        image_path_from = os.path.join(from_path, image_name)
        image_path_to = os.path.join(to_path, 'i{:03d}.jpg'.format(index))
        shutil.copy(image_path_from, image_path_to)


if __name__ == '__main__':
    if_data = dp.get_ifdata(use_csv=True)
    copy_images_indexed(sg_path, if_data)
