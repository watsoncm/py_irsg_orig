import os
import json
from collections import defaultdict

import numpy as np
from sklearn import mixture
from tqdm import tqdm

import data_utils
import gmm_utils
from config import get_config_path
import irsg_core.data_pull as dp

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    sg_path = cfg_data['file_paths']['sg_path']
    data_path = cfg_data['file_paths']['mat_path']
    csv_path = cfg_data['file_paths']['csv_path']


def get_binary_model_data(ifdata, indices, rels=None):
    """Read data for relationship model parameters from CSV files."""
    all_models = defaultdict(list)
    for index in indices:
        image_data = ifdata.vg_data[index]
        for triple in image_data.annotations.binary_triples:
            bbox_pair = (image_data.annotations.objects[triple.subject].bbox,
                         image_data.annotations.objects[triple.object].bbox)
            text_parts = data_utils.get_text_parts(image_data, triple)
            if (rels is not None and text_parts not in rels):
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
    """Train a GMM on a series of bounding box pairs."""
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


def parse_queries(query_path):
    """Read queries from a file."""
    rels = []
    with open(query_path) as f:
        for line in f.read().splitlines():
            parts = line.split()
            raw_query, query_type = parts[:-1], parts[-1]
            query = [part.replace('_', ' ') for part in raw_query]
            if query_type == '(srao)':
                rel = query[1]
                rels.append((query[0], rel, query[3]))
            elif query_type == '(asro)':
                rel = query[2]
                rels.append((query[1], rel, query[3]))
            elif query_type == '(asrao)':
                rel = query[2]
                rels.append((query[1], rel, query[4]))
            general_query = ('*', rel, '*')
            if general_query not in rels:
                rels.append(general_query)
    return rels


if __name__ == '__main__':
    ifdata = dp.get_ifdata(use_csv=True, split='train')
    smol_objs = os.listdir(os.path.join(csv_path, 'datasets', 'psu-small',
                                        'train', 'obj_files'))
    output_paths = [os.path.join(csv_path, 'datasets', *names) for names in
                    (('psu', 'train', 'rel_files'),
                     ('psu', 'val', 'rel_files'),
                     ('psu-small', 'train', 'rel_files'),
                     ('psu-small', 'val', 'rel_files'))]
    splits = ['train', 'val', 'train', 'val']
    query_path = os.path.join(data_path, 'queries.txt')
    smol_rels = parse_queries(query_path)
    rels = [None, None, smol_rels, smol_rels]

    for output_path, split, rels in tqdm(zip(output_paths, splits, rels),
                                         desc='paths'):
        indices = data_utils.get_indices(data_path, split)
        rel_dict = get_binary_model_data(ifdata, indices, rels=rels)
        for text, bbox_pairs in rel_dict.iteritems():
            gmm_data = train_gmm(bbox_pairs)
            gmm_utils.save_gmm_data(text, output_path, *gmm_data)
