import os
import json

import query_viz
import data_utils
import irsg_core.data_pull as dp
import irsg_core.image_fetch_utils as ifu
from config import get_config_path


with open(get_config_path()) as f:
    cfg_data = json.load(f)
    data_path = cfg_data['file_paths']['mat_path']
    out_path = cfg_data['file_paths']['output_path']


def recall_check(queries, if_data, false_negs):
    tp_simple = ifu.get_partial_scene_matches(if_data.vg_data, queries)
    for query_index, image_index in false_negs:
        tp_simple[query_index].append(image_index)
    energy_path = os.path.join(out_path, 'query_energies/')
    geom_energy_path = os.path.join(out_path, 'query_energies_geom/')
    data_simple = [(energy_path, 'factor graph'),
                   (geom_energy_path, 'geometric mean')]
    data_utils.get_recall_values(data_simple, tp_simple, show_plot=True)


if __name__ == '__main__':
    query_path = os.path.join(data_path, 'queries.txt')
    queries = query_viz.generate_queries_from_file(query_path)

    if_data = dp.get_ifdata(use_csv=True)
    batch_path = os.path.join(out_path, 'query_energies/')
    if not os.path.exists(batch_path):
        os.mkdir(batch_path)
        data_utils.generate_energy_data(queries, batch_path, if_data)

    geom_batch_path = os.path.join(out_path, 'query_energies_geom/')
    if not os.path.exists(geom_batch_path):
        os.mkdir(geom_batch_path)
        data_utils.generate_energy_data(queries, geom_batch_path, if_data,
                                        use_geometric=True)

    false_neg_path = os.path.join(data_path, 'false_negs.csv')
    false_negs = data_utils.get_false_negs(false_neg_path)
    recall_check(queries, if_data, false_negs)
