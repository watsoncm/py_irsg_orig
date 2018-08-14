import os
import csv
import json
import glob
from collections import defaultdict

from tqdm import tqdm

from config import get_config_path
import irsg_core.data_pull as dp

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    sg_path = cfg_data['file_paths']['sg_path']
    data_path = cfg_data['file_paths']['mat_path']
    csv_path = cfg_data['file_paths']['csv_path']


def convert_rcnn_data(input_path, output_path):
    bbox_dict = defaultdict(list)
    for data_file in glob.glob(os.path.join(input_path, '*.csv')):
        data_name = os.path.basename(data_file).split('_')[0]
        with open(data_file) as f:
            csv_reader = csv.reader(f, delimiter=' ')
            for line in csv_reader:
                index, score, x1, y1, x2, y2 = [float(value) for value in line]
                x, y, w, h = [int(round(value))
                              for value in (x1, y1, x2 - x1, y2 - y1)]
                bbox_dict[int(index)].append((x, y, w, h, score, -1))

        data_out_path = os.path.join(output_path, data_name)
        if not os.path.exists(data_out_path):
            os.makedirs(data_out_path)
        for index, rows in bbox_dict.iteritems():
            bbox_file = 'irsg_{}.csv'.format(index)
            bbox_path = os.path.join(data_out_path, bbox_file)
            with open(bbox_path, 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(sorted(rows, key=lambda row: row[0]))


if __name__ == '__main__':
    ifdata = dp.get_ifdata(use_csv=True, use_train=True)
    in_paths = [os.path.join(data_path, name) for name in
                ('obj_rcnn_train', 'obj_rcnn_val',
                 'attr_rcnn_train', 'attr_rcnn_val',
                 'obj_rcnn_train_smol', 'obj_rcnn_val_smol',
                 'attr_rcnn_train_smol', 'attr_rcnn_val_smol')]
    out_paths = [os.path.join(csv_path, name) for name in
                 ('obj_files_train', 'obj_files_val',
                  'attr_files_train', 'attr_files_val',
                  'obj_files_train_smol', 'obj_files_val_smol',
                  'attr_files_train_smol', 'attr_files_val_smol')]
    for in_path, out_path in tqdm(zip(in_paths, out_paths)):
        convert_rcnn_data(in_path, out_path)
