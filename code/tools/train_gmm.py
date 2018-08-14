import os
import json
from collections import defaultdict

import numpy as np
from sklearn import mixture
from tqdm import tqdm

import data_utils
from config import get_config_path
import irsg_core.data_pull as dp

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    sg_path = cfg_data['file_paths']['sg_path']
    data_path = cfg_data['file_paths']['mat_path']
    csv_path = cfg_data['file_paths']['csv_path']


def get_binary_model_data(ifdata, indices, objs=None):
    all_models = defaultdict(list)
    for index in indices:
        image_data = ifdata.vg_data[index]
        for triple in image_data.annotations.binary_triples:
            bbox_pair = (image_data.annotations.objects[triple.subject].bbox,
                         image_data.annotations.objects[triple.object].bbox)
            text_parts = data_utils.get_text_parts(image_data, triple)
            if (objs is not None
                    and text_parts[0] not in objs
                    and text_parts[2] not in objs):
                continue
            all_models[text_parts].append(bbox_pair)

    models, small_models = defaultdict(list), defaultdict(list)
    for text_parts, bbox_pairs in all_models.iteritems():
        text = '_'.join(text_parts).replace(' ', '_')
        generic_text = '*_{}_*'.format(text_parts[1].replace(' ', '_'))
        rel_model_names = ifdata.relationship_models.keys()
        if text in rel_model_names:
            models[text] = bbox_pairs
        elif generic_text in rel_model_names:
            small_models[generic_text].extend(bbox_pairs)

    models.update(small_models)
    return models


def train_gmm(bbox_pairs):
    subs, objs = zip(*bbox_pairs)
    sub_x, sub_y, sub_w, sub_h = zip(*[(sub.x, sub.y, sub.w, sub.h)
                                       for sub in subs])
    sub_x, sub_y, sub_w, sub_h = [np.array(value, dtype=float)
                                  for value in (sub_x, sub_y, sub_w, sub_h)]
    obj_x, obj_y, obj_w, obj_h = zip(*[(obj.x, obj.y, obj.w, obj.h)
                                       for obj in objs])
    obj_x, obj_y, obj_w, obj_h = [np.array(value, dtype=float)
                                  for value in (obj_x, obj_y, obj_w, obj_h)]
    sub_center_x, sub_center_y = sub_x + 0.5 * sub_w, sub_y + 0.5 * sub_h
    obj_center_x, obj_center_y = obj_x + 0.5 * obj_w, obj_y + 0.5 * obj_h
    rel_center_x = (sub_center_x - obj_center_x) / sub_w
    rel_center_y = (sub_center_y - obj_center_y) / sub_h
    rel_width, rel_height = obj_w / sub_w, obj_h / sub_h
    features = np.column_stack((rel_center_x, rel_center_y,
                                rel_width, rel_height))

    while features.shape[0] < 3:
        features = np.vstack((features, features[-1, :]))

    gmm = mixture.GaussianMixture(n_components=3)
    gmm.fit(features)
    return gmm.weights_, gmm.means_, gmm.covariances_


def save_gmm_data(text, path, weights, means, covariances):
    gmm_path = os.path.join(path, text)
    if not os.path.exists(gmm_path):
        os.makedirs(gmm_path)
    weights_path = os.path.join(gmm_path, 'weights.csv')
    means_path = os.path.join(gmm_path, 'means.csv')
    covar_paths = [os.path.join(gmm_path, 'covar{}.csv'.format(i))
                   for i in range(len(covariances))]
    np.savetxt(weights_path, weights)
    np.savetxt(means_path, means)
    for covar_path, covariance in zip(covar_paths, covariances):
        np.savetxt(covar_path, covariance)


if __name__ == '__main__':
    ifdata = dp.get_ifdata(use_csv=True, use_train=True)
    with open(os.path.join()) as f:
        smol_objs = f.read().splitlines()

    output_paths = [os.path.join(csv_path, name) for name in
                    ('rel_files_train', 'rel_files_val',
                     'rel_files_train_smol', 'rel_files_val_smol')]
    splits = ['train', 'val', 'train', 'val']
    objs = [None, None, smol_objs, smol_objs]

    for output_path, split, objs in tqdm(zip(output_paths, splits, objs)):
        indices = data_utils.get_indices(data_path, split)
        rel_dict = get_binary_model_data(ifdata, indices, objs=objs)
        for text, bbox_pairs in rel_dict.iteritems():
            gmm_data = train_gmm(bbox_pairs)
            save_gmm_data(text, output_path, *gmm_data)
