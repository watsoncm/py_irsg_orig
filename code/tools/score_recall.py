import os
import json

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


def recall_check(queries, if_data, false_negs):
    """Compute recalls for a given set of queries and plot."""
    tp_simple = data_utils.get_partial_query_matches(if_data.vg_data, queries)
    for query_index, image_index in false_negs:
        tp_simple[query_index].append(image_index)
    energy_path = os.path.join(out_path, 'query_energies/')
    geom_energy_path = os.path.join(out_path, 'query_energies_geom/')
    weighted_energy_path = os.path.join(out_path, 'query_energies_weighted/')
    data_simple = [(energy_path, 'vanilla IRSG'),
                   (geom_energy_path, 'geometric mean'),
                   (weighted_energy_path, 'weighted IRSG')]
    data_utils.get_single_image_recall_values(
        data_simple, tp_simple, len(if_data.vg_data), show_plot=True)


if __name__ == '__main__':
    query_path = os.path.join(data_path, 'queries.txt')
    queries = query_viz.generate_queries_from_file(query_path)

    if_data = dp.get_ifdata(use_csv=True)
    # batch_path = os.path.join(out_path, 'query_energies/')
    # if not os.path.exists(batch_path):
    #     os.mkdir(batch_path)
    #     image_query_data.generate_energy_data(queries, batch_path, if_data)

    # geom_batch_path = os.path.join(out_path, 'query_energies_geom/')
    # if not os.path.exists(geom_batch_path):
    #     os.mkdir(geom_batch_path)
    #     image_query_data.generate_energy_data(
    #         queries, geom_batch_path, if_data, use_geometric=True)

    weighted_batch_path = os.path.join(out_path, 'query_energies_weighted/')
    if not os.path.exists(weighted_batch_path):
        os.mkdir(weighted_batch_path)
        image_query_data.generate_energy_data(
            queries, weighted_batch_path, if_data,
            pred_weight=PRED_WEIGHT)

    false_neg_path = os.path.join(data_path, 'false_negs.csv')
    false_negs = data_utils.get_false_negs(false_neg_path)
    recall_check(queries, if_data, false_negs)
