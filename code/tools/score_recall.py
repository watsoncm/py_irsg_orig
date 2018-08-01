import csv
import os
import json

import numpy as np
from tqdm import tqdm

import query_viz
import irsg_core.data_pull as dp
import irsg_core.image_fetch_utils as ifu
import irsg_core.image_fetch_plot as ifp
from config import get_config_path


with open(get_config_path()) as f:
    cfg_data = json.load(f)
    data_path = cfg_data['file_paths']['mat_path']
    out_path = cfg_data['file_paths']['output_path']


def generate_energy_data(queries, path, if_data, use_geometric=False):
    for query_id, query in tqdm(enumerate(queries), total=len(queries)):
        energy_list = []
        for image_id in tqdm(range(len(if_data.vg_data))):
            iqd = query_viz.ImageQueryData(query, query_id, image_id,
                                           if_data, compute_gt=False)
            iqd.compute_potential_data(use_relationships=not use_geometric)
            energy = (np.sqrt(iqd.model_sub_pot * iqd.model_obj_pot)
                      if use_geometric else iqd.model_total_pot)
            energy_list.append(energy)

        energy_name = 'q{:03d}_energy_values.csv'.format(query_id)
        energy_path = os.path.join(path, energy_name)
        with open(energy_path, 'wb'):
            csv_writer = csv.writer(file)
            csv_writer.writerow(("image_ix", "energy"))
            for image_id, energy in enumerate(energy_list):
                csv_writer.writerow((image_id, energy))


def recall_check(queries, if_data, false_negs, do_holdout=False, x_limit=100):
    tp_simple = ifu.get_partial_scene_matches(if_data.vg_data, queries)
    for query_index, image_index in false_negs:
        tp_simple[query_index].append(image_index)
    energy_path = os.path.join(out_path, 'query_energies/')
    geom_energy_path = os.path.join(out_path, 'geom_query_energies/')
    data_simple = [(energy_path, 'factor graph'),
                   (geom_energy_path, 'geometric mean')]
    ifp.r_at_k_plot_simple(data_simple, tp_simple,
                           do_holdout=do_holdout,
                           x_limit=x_limit)


def get_false_negs(path):
    with open(path) as f:
        csv_reader = csv.reader(f)
        false_negs = [(int(query_index), int(image_index))
                      for query_index, image_index in list(csv_reader)[1:]]
    return false_negs


if __name__ == '__main__':
    query_path = os.path.join(data_path, 'queries.txt')
    queries = query_viz.generate_queries_from_file(query_path)

    if_data = dp.get_ifdata(use_csv=True)
    batch_path = os.path.join(out_path, 'query_energies/')
    if not os.path.exists(batch_path):
        os.mkdir(batch_path)
        generate_energy_data(queries, batch_path, if_data)

    geom_batch_path = os.path.join(out_path, 'query_energies_geom/')
    if not os.path.exists(geom_batch_path):
        os.mkdir(geom_batch_path)
        generate_energy_data(queries, geom_batch_path, if_data,
                             use_geometric=True)

    false_neg_path = os.path.join(data_path, 'false_negs.csv')
    false_negs = get_false_negs(false_neg_path)
    recall_check(queries, if_data, false_negs)
