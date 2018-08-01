import os
import shutil
import json

import data_pull as dp

with open('config.json') as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']
    data_path = cfg_data['file_paths']['mat_path']
    sg_path = cfg_data['file_paths']['sg_path']


def generate_rcnn_data(ifdata):
    dataset_path = os.path.join(out_path, 'IRSG_RCNN')
    anno_path = os.path.join(dataset_path, 'Annotations')
    image_path = os.path.join(dataset_path, 'Images')
    set_path = os.path.join(dataset_path, 'ImageSets')

    for path in (anno_path, image_path, set_path):
        if not os.path.exists(path):
            os.makedirs(path)

    train_indices, test_indices = [], []
    for image_index in range(len(ifdata.vg_data)):
        ifdata.configure(image_index, None)
        shutil.copy(self.image_filename, image_path)
        


if __name__ == '__main__':
    ifdata = dp.get_ifdata(use_csv=True)
    generate_rcnn_data(ifdata)
