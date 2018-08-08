import os
import json

import numpy as np
from tqdm import tqdm

import query_viz
import data_utils
import irsg_core.data_pull as dp
import irsg_core.image_fetch_utils as ifu
from config import get_config_path

NUM_XVALS = 5
K = 10


with open(get_config_path()) as f:
    cfg_data = json.load(f)
    data_path = cfg_data['file_paths']['mat_path']
    out_path = cfg_data['file_paths']['output_path']


def run_cross_val_test(if_data, queries, path, index,
                       obj_weight, attr_weight, pred_weight):
    iqd_params = {'obj_weight': obj_weight,
                  'attr_weight': attr_weight,
                  'pred_weight': pred_weight}
    data_utils.generate_energy_data(queries, path, if_data,
                                    iqd_params=iqd_params)
    data_simple = [(path, 'xval {} energies'.format(index))]
    false_neg_path = os.path.join(data_path, 'false_negs.csv')
    false_negs = data_utils.get_false_negs(false_neg_path)
    tp_simple = ifu.get_partial_scene_matches(if_data.vg_data, queries)
    for query_index, image_index in false_negs:
        tp_simple[query_index].append(image_index)
    recalls = data_utils.get_recall_values(data_simple, tp_simple,
                                           len(if_data.vg_data))[0]
    return recalls


def cross_validate(if_data, queries, k_val, num_xvals):
    query_format = 'query_energies_xval_{}/'
    result_format = ('obj_weight: {}\nattr_weight: {}\n'
                     'pred_weight: {}\nR@{}: {}')
    weights = np.array([np.random.rand(3) for _ in range(num_xvals)])
    weights /= np.sum(weights, axis=1)[:, np.newaxis]
    results = []
    recalls_at_k = []
    for index, weights in tqdm(enumerate(weights), total=len(weights),
                               desc='xval'):
        obj_weight, attr_weight, pred_weight = weights
        batch_path = os.path.join(out_path, query_format.format(index))
        if not os.path.exists(batch_path):
            os.mkdir(batch_path)
        recalls = run_cross_val_test(if_data, queries, batch_path,
                                     index, *weights)
        recalls_at_k.append(recalls[k_val])
        result = result_format.format(weights[0], weights[1],
                                      weights[2], K, recalls[K])
        results.append(result)
        recalls_at_k.append(recalls[k_val])
        with open(os.path.join(batch_path, 'results.txt'), 'w') as f:
            f.write(result)

    with open(os.path.join(out_path, 'xval_results.txt'), 'w') as f:
        f.write(results[np.argmax(recalls_at_k)])


if __name__ == '__main__':
    if_data = dp.get_ifdata(use_csv=True, use_train=True)
    query_path = os.path.join(data_path, 'queries.txt')
    queries = query_viz.generate_queries_from_file(query_path)
    cross_validate(if_data, queries, K, NUM_XVALS)