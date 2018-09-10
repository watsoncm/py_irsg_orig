import os
import json

import query_viz
import data_utils
import image_query_data
import irsg_core.data_pull as dp
from config import get_config_path


with open(get_config_path()) as f:
    cfg_data = json.load(f)
    data_path = cfg_data['file_paths']['mat_path']
    out_path = cfg_data['file_paths']['output_path']


def iou_check(queries, if_data):
    tp_simple = data_utils.get_partial_query_matches(if_data.vg_data, queries)
    energy_path = os.path.join(out_path, 'query_ious/')
    geom_energy_path = os.path.join(out_path, 'query_ious_geom/')
    data_simple = [(energy_path, 'factor graph'),
                   (geom_energy_path, 'geometric mean')]
    data_utils.get_iou_recall_values(
        data_simple, tp_simple, len(if_data.vg_data), show_plot=True)


if __name__ == '__main__':
    query_path = os.path.join(data_path, 'queries.txt')
    queries = query_viz.generate_queries_from_file(query_path)

    if_data = dp.get_ifdata(use_csv=True)
    batch_path = os.path.join(out_path, 'query_ious/')
    if not os.path.exists(batch_path):
        os.mkdir(batch_path)
        image_query_data.generate_iou_data(queries, batch_path, if_data)

    geom_batch_path = os.path.join(out_path, 'query_ious_geom/')
    if not os.path.exists(geom_batch_path):
        os.mkdir(geom_batch_path)
        image_query_data.generate_iou_data(
            queries, geom_batch_path, if_data, use_geometric=True)

    iou_check(queries, if_data)
