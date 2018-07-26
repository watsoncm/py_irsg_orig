import csv
import os
import json
from collections import Counter

import numpy as np

import data_pull as dp

with open('config.json') as f:
    cfg_data = json.load(f)
    csv_path = cfg_data['file_paths']['csv_path']


def get_obj_attr_counts(if_data):
    obj_count = Counter()
    attr_count = Counter()
    for image in if_data.vg_data:
        for triple in image.annotations.unary_triples:
            attr_name = triple.text[2]
            attr_count[attr_name] += 1
        for obj in image.annotations.objects:
            obj_name = np.array(obj.names).reshape(-1)[0]
            obj_count[obj_name] += 1
    return obj_count, attr_count


def write_obj_attr_counts(obj_count, attr_count, train_objs,
                          train_attrs, path):
    obj_path = os.path.join(path, 'obj_counts.csv')
    attr_path = os.path.join(path, 'attr_counts.csv')
    for csv_path, count, train_vals in ((obj_path, obj_count, train_objs),
                                        (attr_path, attr_count, train_attrs)):
        with open(csv_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(('name', 'count', 'is_counted'))
            for obj_attr, count in count.most_common():
                csv_writer.writerow((obj_attr, count, obj_attr in train_vals))


def count_to_threshold_list(count, thresh):
    return [obj for obj, obj_count in count.most_common()
            if obj_count >= thresh]


if __name__ == '__main__':
    if_data_test = dp.get_ifdata(use_csv=True)
    if_data_train = dp.get_ifdata(use_csv=True, use_train=True)
    test_obj_count, test_attr_count = get_obj_attr_counts(if_data_test)
    train_obj_count, train_attr_count = get_obj_attr_counts(if_data_train)
    train_objs = count_to_threshold_list(train_obj_count, 50)
    train_attrs = count_to_threshold_list(train_attr_count, 50)
    write_obj_attr_counts(test_obj_count, test_attr_count, train_objs,
                          train_attrs, csv_path)
