import os
import json
import shutil
from tqdm import tqdm

import numpy as np

import query_viz
import data_utils
import irsg_core.data_pull as dp
import irsg_core.image_fetch_utils as ifu
from image_query_data import ImageQueryData
from config import get_config_path

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']
    data_path = cfg_data['file_paths']['mat_path']


def generate_situations(queries, situation_path, if_data, if_data_test,
                        false_negs):
    tp_data = ifu.get_partial_scene_matches(if_data.vg_data, queries)
    tp_data_test = ifu.get_partial_scene_matches(if_data_test.vg_data, queries)
    for query_index, image_index in false_negs:
        tp_data_test[query_index].append(image_index)

    query_data = zip(queries, tp_data, tp_data_test)
    for index, (query, train_tps, test_tps) in tqdm(
            enumerate(query_data), total=len(queries), desc='queries'):
        iqd = ImageQueryData(query, index, 0, if_data)
        query_name = iqd.get_query_text().replace(' ', '-')
        query_path = os.path.join(situation_path, query_name)
        pos_path = os.path.join(query_path, query_name)
        neg_path = os.path.join(query_path, '{}-negative'.format(query_name))
        test_path = os.path.join(query_path, '{}-test'.format(query_name))
        for path in (pos_path, neg_path, test_path):
            if not os.path.exists(path):
                os.makedirs(path)
        for tp_index in tqdm(train_tps, desc='train images'):
            train_iqd = ImageQueryData(query, index, tp_index, if_data)
            if_data.configure(tp_index, query.annotations)
            try:
                box_pairs = train_iqd.get_image_boxes(return_all=True)
            except ValueError:  # small issue with some train queries
                continue
            sub_box, obj_box = [list(pairs) for pairs in
                                box_pairs[np.random.choice(len(box_pairs))]]
            image_data = if_data.vg_data[tp_index]
            json_data = {'im_w': image_data.image_width,
                         'im_h': image_data.image_height,
                         'objects': [
                             {'desc': train_iqd.query_sub,
                              'box_xywh': sub_box},
                             {'desc': train_iqd.query_obj,
                              'box_xywh': obj_box}]}
            json_name = os.path.splitext(os.path.basename(
                if_data.image_filename))[0] + '.json'
            json_path = os.path.join(pos_path, json_name)
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
            shutil.copy(if_data.image_filename, pos_path)

        for index in tqdm(range(len(
                if_data_test.vg_data)), desc='test images'):
            if_data.configure(index, query.annotations)
            target_path = test_path if index in test_tps else neg_path
            shutil.copy(if_data.image_filename, target_path)


if __name__ == '__main__':
    if_data = dp.get_ifdata('psu', split='train', use_csv=True)
    if_data_test = dp.get_ifdata('stanford', split='test', use_csv=True)
    path = os.path.join(data_path, 'queries.txt')
    queries = query_viz.generate_queries_from_file(path)
    query_path = os.path.join(out_path, 'situations')
    false_neg_path = os.path.join(data_path, 'false_negs.csv')
    false_negs = data_utils.get_false_negs(false_neg_path)
    generate_situations(queries, query_path, if_data, if_data_test, false_negs)
