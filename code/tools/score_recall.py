import os
import json

import numpy as np

import query_viz
import data_utils
import image_query_data
import irsg_core.data_pull as dp
from config import get_config_path

PRED_WEIGHT = 0.1


with open(get_config_path()) as f:
    cfg_data = json.load(f)
    data_path = cfg_data['file_paths']['mat_path']
    out_path = cfg_data['file_paths']['output_path']
    csv_path = cfg_data['file_paths']['csv_path']


def recall_check(queries, if_data, false_negs, situate_recalls):
    """Compute recalls for a given set of queries and plot."""
    tp_simple = data_utils.get_partial_query_matches(if_data.vg_data, queries)
    for query_index, image_index in false_negs:
        tp_simple[query_index].append(image_index)
    energy_path = os.path.join(out_path, 'query_energies/')
    geom_energy_path = os.path.join(out_path, 'query_energies_geom/')
    true_geom_energy_path = os.path.join(out_path, 'query_energies_true_geom/')
    weighted_energy_path = os.path.join(out_path, 'query_energies_weighted/')
    rcnn_energy_path = os.path.join(out_path, 'query_energies_rcnn/')
    data_simple = [(energy_path, 'vanilla IRSG'),
                   (geom_energy_path, 'geometric mean on potentials'),
                   (true_geom_energy_path, 'true geometric mean'),
                   (weighted_energy_path, 'weighted IRSG'),
                   (rcnn_energy_path, 'RCNN-weighted IRSG')]

    data_utils.get_single_image_recall_values(
        data_simple, tp_simple, len(if_data.vg_data), show_plot=True,
        situate_recalls=situate_recalls)


def get_situate_recalls(num_queries, situate_path):
    recalls = [[] for _ in range(num_queries)]
    for query_index in range(num_queries):
        csv_path = os.path.join(
            situate_path, 'q{:03d}_recalls.csv'.format(query_index))
        try:
            recall_array = np.loadtxt(csv_path, delimiter=',').reshape(-1)
        except IOError:
            continue
        recalls[query_index] = list(recall_array)
    return recalls


if __name__ == '__main__':
    query_path = os.path.join(data_path, 'queries.txt')
    queries = query_viz.generate_queries_from_file(query_path)

    if_data = dp.get_ifdata(use_csv=True)
    batch_path = os.path.join(out_path, 'query_energies/')
    if not os.path.exists(batch_path):
        os.mkdir(batch_path)
        image_query_data.generate_energy_data(queries, batch_path, if_data)

    geom_batch_path = os.path.join(out_path, 'query_energies_geom/')
    if not os.path.exists(geom_batch_path):
        os.mkdir(geom_batch_path)
        image_query_data.generate_energy_data(
            queries, geom_batch_path, if_data, use_alt_geom=True)

    true_geom_batch_path = os.path.join(out_path, 'query_energies_true_geom/')
    if not os.path.exists(true_geom_batch_path):
        os.mkdir(true_geom_batch_path)
        image_query_data.generate_energy_data(
            queries, true_geom_batch_path, if_data, use_geometric=True)

    weighted_batch_path = os.path.join(out_path, 'query_energies_weighted/')
    if not os.path.exists(weighted_batch_path):
        os.mkdir(weighted_batch_path)
        image_query_data.generate_energy_data(
            queries, weighted_batch_path, if_data,
            pred_weight=PRED_WEIGHT)

    rcnn_batch_path = os.path.join(out_path, 'query_energies_rcnn/')
    rcnn_path = os.path.join(csv_path, 'rcnn_weights.csv')
    if not os.path.exists(rcnn_batch_path):
        os.mkdir(rcnn_batch_path)
        image_query_data.generate_energy_data(
            queries, rcnn_batch_path, if_data,
            rcnn_weights=data_utils.get_rcnn_weights(rcnn_path))

    situate_path = os.path.join(data_path, 'situate_recalls')
    situate_recalls = get_situate_recalls(len(queries), situate_path)

    false_neg_path = os.path.join(data_path, 'false_negs.csv')
    false_negs = data_utils.get_false_negs(false_neg_path)
    recall_check(queries, if_data, false_negs, situate_recalls)
