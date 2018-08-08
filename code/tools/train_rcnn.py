import os
import csv
import json
import glob
from collections import defaultdict

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
    obj_path = os.path.join(data_path, 'obj_rcnn_data')
    obj_out_path = os.path.join(csv_path, 'obj_data_custom')
    attr_path = os.path.join(data_path, 'attr_rcnn_data')
    attr_out_path = os.path.join(csv_path, 'attr_data_custom')
    convert_rcnn_data(obj_path, obj_out_path)
    convert_rcnn_data(obj_path, attr_out_path)
