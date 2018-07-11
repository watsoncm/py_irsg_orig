import csv
import os
import json

import numpy as np
from tqdm import tqdm

import data_pull as dp

with open('config.json') as f:
    cfg_data = json.load(f)
    csv_path = cfg_data['file_paths']['csv_path']


def write_scores(out_path, detections):
    """Write object/attribute detection scores to CSV."""
    neg_col = -np.ones((detections.shape[0], 1))  # for compatibility
    detections_full = np.hstack((detections, neg_col))
    np.savetxt(out_path, detections_full,
               fmt='%i,%i,%i,%i,%.6f,%i')


def transfer_scores(ifdata, root_path, is_attr=False):
    """Transfer scores from Matlab format to CSV."""
    desc = 'attr' if is_attr else 'objects'
    for image_idx in tqdm(np.arange(ifdata.vg_data.size), desc=desc):
        ifdata.configure(image_idx, None)
        detection_dict = (ifdata.attribute_detections if is_attr else
                          ifdata.object_detections)
        csv_name = 'irsg_{}.csv'.format(image_idx)
        for name, result in tqdm(detection_dict.iteritems(),
                                 total=len(detection_dict)):
            name = name[4:]
            path = os.path.join(root_path, name)
            if not os.path.exists(path):
                os.mkdir(path)
            img_csv_path = os.path.join(path, csv_name)
            write_scores(img_csv_path, result)


def transfer_class_to_idx(ifdata, path):
    """Transfer class to index dictionary from Matlab format to CSV."""
    class_to_index = ifdata.potentials_data['potentials_s'].class_to_idx
    serial = class_to_index.serialization
    with open(path, 'w') as f:
        csv_writer = csv.writer(f)
        for key, val in tqdm(zip(serial.keys, serial.values),
                             desc='class/index'):
            csv_writer.writerow((key, val))


def generate_image_csvs(ifdata, root_path):
    """Generate a single CSV for each image with all unary potential data."""
    image_idx_arrays = []
    name_arrays = []
    box_idx_arrays = []
    value_arrays = []
    dtype = [('name', 'U24'),
             ('box_idx', 'i4'),
             ('b0', 'f4'),
             ('b1', 'f4'),
             ('b2', 'f4'),
             ('b3', 'f4'),
             ('score', 'U10')]


    for image_idx in tqdm(np.arange(ifdata.vg_data.size), desc='images'):
        csv_name = 'irsg_{}.csv'.format(image_idx)
        csv_path = os.path.join(root_path, csv_name)
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            ifdata.configure(image_idx, None)
            for desc, detections in [('attrs', ifdata.attribute_detections),
                                     ('objs', ifdata.object_detections)]:
                for name, values in tqdm(detections.iteritems(),
                                         total=len(detections), desc=desc):
                    output_array = np.empty(values.shape[0], dtype=dtype)
                    output_array['name'] = np.full(values.shape[0], name, dtype='U25')
                    output_array['box_idx'] = np.arange(values.shape[0])
                    for i in range(4):
                        output_array['b' + str(i)] = values[:, i]
                    output_array['score'] = np.around(values[:, 4], decimals=6)
                    writer.writerows(output_array)


def convert_all_to_csv(by_image=False):
    """Converts all Matlab files into corresponding CSV files."""
    ifdata = dp.get_ifdata()
    if by_image:
        image_csv_path = os.path.join(csv_path, 'image_files')
        if not os.path.exists(image_csv_path):
            os.mkdir(image_csv_path)
        generate_image_csvs(ifdata, image_csv_path)
    else:
        obj_path = os.path.join(csv_path, 'obj_files')
        attr_path = os.path.join(csv_path, 'attr_files')
        class_to_idx_path = os.path.join(csv_path, 'class_to_idx.csv')

        for path in (obj_path, attr_path):
            if not os.path.exists(path):
                os.mkdir(path)

        transfer_scores(ifdata, obj_path)
        transfer_scores(ifdata, attr_path, is_attr=True)
        transfer_class_to_idx(ifdata, class_to_idx_path)


if __name__ == '__main__':
    convert_all_to_csv()
