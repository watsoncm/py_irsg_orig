import re
import csv
import os
import glob
import json

import numpy as np
from tqdm import tqdm
from itertools import product
from sklearn import linear_model

import data_utils
import irsg_core.data_pull as dp
from image_query_data import ImageQueryData
from config import get_config_path

NUM_NEGS = 20

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    sg_path = cfg_data['file_paths']['sg_path']
    data_path = cfg_data['file_paths']['mat_path']
    csv_path = cfg_data['file_paths']['csv_path']


def test_bbox_class(image_index, bbox, class_name, ifdata):
    objects = ifdata.vg_data[image_index].annotations.objects
    for obj in objects:
        obj_name = np.array(obj.names).reshape(-1)[0]
        obj_bbox = (obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h)
        if (ImageQueryData.get_iou(bbox, obj_bbox) > 0.5
                and obj_name == class_name):
            return True
    return False


def get_rcnn_data(path, ifdata, image_indices, desc=None):
    data = {}
    for class_name in tqdm(os.listdir(path), desc=desc):
        class_path = os.path.join(path, class_name)
        scores = []
        gts = []
        for data_file in tqdm(glob.glob(os.path.join(class_path, '*.csv'))):
            data_name = os.path.basename(data_file)
            image_index = int(re.findall('\d+', data_name)[0])
            if image_index not in image_indices:
                continue  # no cheating
            with open(data_file) as f:
                csv_reader = csv.reader(f)
                for line in csv_reader:
                    bbox = [int(value) for value in line[:4]]
                    score = float(line[4])
                    gt = test_bbox_class(image_index, bbox,
                                         class_name, ifdata)
                    gts.append(1 if gt else 0)
                    scores.append(score)
        data[class_name] = (scores, gts)
    return data


def get_gmm_data(path, ifdata, image_indices, negs_per_image=10, desc=None):
    data = {}
    for rel_name in tqdm(os.listdir(path), desc=desc):
        rel_path = os.path.join(path, rel_name)
        scores = []
        gts = []
        weights = np.loadtxt(os.path.join(rel_path, 'weights.csv'))
        means = np.loadtxt(os.path.join(rel_path, 'means.csv'))
        covar_dict = {}
        for data_name in os.listdir(rel_path):
            if 'covar' in data_name:
                covar_index = int(re.findall('\d+', data_name)[0])
                covar_path = os.path.join(rel_path, data_name)
                covar_dict[covar_index] = np.loadtxt(covar_path)
        covars = np.array([covar_dict[index]
                           for index in range(len(covar_dict))])
        gmm_params = (weights, means, covars)
        for image_index in tqdm(image_indices, desc='images'):
            image_data = ifdata.vg_data[image_index]
            gt_relations = []
            for triple in image_data.annotations.binary_triples:
                text_parts = data_utils.get_text_parts(image_data, triple)
                triple_text = '_'.join(text_parts).replace(' ', '_')
                pred_text = triple.predicate.replace(' ', '_')
                if (triple_text == rel_name or
                        ('*' in rel_name and rel_name[2:-2] == pred_text)):
                    gt_relations.append((triple.subject, triple.object))
            bboxes = [(float(obj.bbox.x), float(obj.bbox.y),
                       float(obj.bbox.w), float(obj.bbox.h))
                      for obj in image_data.annotations.objects]

            all_pairs = product(range(len(bboxes)), repeat=2)
            all_neg_pairs = [pair for pair in all_pairs
                             if pair not in gt_relations]
            num_negs = min((len(all_neg_pairs), negs_per_image))
            neg_indices = np.random.choice(len(all_neg_pairs),
                                           size=num_negs,
                                           replace=False)
            neg_pairs = [all_neg_pairs[index] for index in neg_indices]
            pairs = gt_relations + list(neg_pairs)
            gts.extend([1] * len(gt_relations) + [0] * len(neg_pairs))
            for sub_index, obj_index in pairs:
                bbox_sub, bbox_obj = bboxes[sub_index], bboxes[obj_index]
                score = ImageQueryData.score_box_pair(bbox_sub, bbox_obj,
                                                      *gmm_params)[0]
                scores.append(score)
        data[rel_name] = (scores, gts)
    return data


def save_platt_params(data, path):
    for name, (scores, gts) in data.iteritems():
        lr = linear_model.LogisticRegression()
        score_array, gt_array = np.array(scores).reshape(-1, 1), np.array(gts)
        try:
            lr.fit(score_array, gt_array)
            coef, intercept = lr.coef_[0][0], lr.intercept_[0]
        except ValueError:
            coef, intercept = 0.0, float(gts[0])
        platt_path = os.path.join(path, name, 'platt.csv')
        with open(platt_path, 'w') as f:
            f.write('{},{}'.format(coef, intercept))


if __name__ == '__main__':
    ifdata = dp.get_ifdata(use_csv=True, use_train=True)
    indices = data_utils.get_indices(data_path, 'train')

    obj_train_path = os.path.join(csv_path, 'obj_files_train')
    attr_train_path = os.path.join(csv_path, 'attr_files_train')
    rel_train_path = os.path.join(csv_path, 'rel_files_train')
    obj_val_path = os.path.join(csv_path, 'obj_files_val')
    attr_val_path = os.path.join(csv_path, 'attr_files_val')
    rel_val_path = os.path.join(csv_path, 'rel_files_val')

    obj_data = get_rcnn_data(obj_train_path, ifdata, indices, desc='objs')
    save_platt_params(obj_data, obj_val_path)

    attr_data = get_rcnn_data(attr_train_path, ifdata, indices, desc='attrs')
    save_platt_params(attr_data, attr_val_path)

    rel_data = get_gmm_data(rel_train_path, ifdata, indices, desc='rels')
    save_platt_params(rel_data, rel_val_path)
