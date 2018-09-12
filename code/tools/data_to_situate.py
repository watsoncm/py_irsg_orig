import os
import csv
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
    all_neg_path = os.path.join(situation_path, 'negative-images')
    for index, (query, train_tps, test_tps) in tqdm(
            enumerate(query_data), total=len(queries), desc='queries'):
        iqd = ImageQueryData(query, index, 0, if_data)
        query_name = iqd.get_query_text().replace(' ', '-')
        query_path = os.path.join(situation_path, query_name)
        pos_path = os.path.join(query_path, query_name)
        neg_path = os.path.join(query_path, '{}-negative'.format(query_name))
        test_path = os.path.join(query_path, '{}-test'.format(query_name))
        for path in (pos_path, all_neg_path, test_path):
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
        for tp_index in tqdm(test_tps, desc='test images'):
            if_data_test.configure(tp_index, query.annotations)
            shutil.copy(if_data_test.image_filename, test_path)
        rel_path = os.path.relpath(
            all_neg_path, start=os.path.dirname(neg_path))
        os.symlink(rel_path, neg_path)

    for tn_index in tqdm(
            [index for index in range(len(if_data_test.vg_data))
             if all([index not in test_tps for test_tps in tp_data_test])]):
        if_data_test.configure(tn_index, query.annotations)
        shutil.copy(if_data_test.image_filename, all_neg_path)


def generate_situate_rnns(queries, situation_path, situation_rcnn_path,
                          if_data_test):
    for query_id, query in tqdm(enumerate(queries), total=len(queries),
                                desc='queries'):
        iqd = ImageQueryData(query, query_id, 0, if_data)
        query_name = iqd.get_query_text().replace(' ', '-')
        query_path = os.path.join(situation_path, query_name,
                                  '{}-test'.format(query_name))
        neg_path = os.path.join(situation_path, 'negative-images')
        query_rcnn_path = os.path.join(situation_rcnn_path, query_name)
        images = os.listdir(query_path) + os.listdir(neg_path)
        for image_id in tqdm(range(len(if_data_test.vg_data))):
            if_data_test.configure(image_id, query.annotations)
            image_name = os.path.basename(if_data_test.image_filename)
            if image_name not in images:
                continue
            rcnn_name = '{}.csv'.format(
                os.path.splitext(image_name)[0])
            obj_names = [obj.names for obj in query.annotations.objects]
            for obj_name in obj_names:
                long_name = 'obj:{}'.format(obj_name)
                obj_id = if_data_test.potentials_data[
                    'class_to_idx'][long_name] - 1
                box_score_data = if_data_test.potentials_data
                boxes = box_score_data['boxes'][image_id][obj_id]
                scores = box_score_data['scores'][image_id][obj_id]
                obj_dir = os.path.join(query_rcnn_path, obj_name)
                rcnn_path = os.path.join(obj_dir, rcnn_name)
                print(obj_dir)
                if not os.path.exists(obj_dir):
                    print('making objdir')
                    os.makedirs(obj_dir)
                with open(rcnn_path, 'w') as f:
                    writer = csv.writer(f)
                    for box, score in zip(boxes, scores):
                        writer.writerow(list(box) + [score])


if __name__ == '__main__':
    if_data = dp.get_ifdata('psu', split='train', use_csv=True)
    if_data_test = dp.get_ifdata('stanford', split='test', use_csv=True)
    path = os.path.join(data_path, 'queries.txt')
    queries = query_viz.generate_queries_from_file(path)
    query_path = os.path.join(out_path, 'situations')
    query_rcnn_path = os.path.join(out_path, 'situations_rcnn')
    false_neg_path = os.path.join(data_path, 'false_negs.csv')
    false_negs = data_utils.get_false_negs(false_neg_path)
    # generate_situations(queries, query_path, if_data, if_data_test, false_negs)
    generate_situate_rnns(queries, query_path, query_rcnn_path, if_data_test)
