import os
import csv
import json

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from image_query_data import ImageQueryData
from config import get_config_path

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']


def generate_energy_data(queries, path, if_data, use_geometric=False,
                         iqd_params=None):
    for query_id, query in tqdm(list(enumerate(queries)),
                                total=len(queries)):
        energy_list = []
        for image_id in tqdm(range(len(if_data.vg_data))):
            if iqd_params is None:
                iqd_params = {}
            iqd = ImageQueryData(query, query_id, image_id,
                                 if_data, compute_gt=False,
                                 **iqd_params)
            iqd.compute_potential_data(use_relationships=not use_geometric)
            energy = (np.sqrt(iqd.model_sub_pot * iqd.model_obj_pot)
                      if use_geometric else iqd.model_total_pot)
            energy_list.append(energy)

        energy_name = 'q{:03d}_energy_values.csv'.format(query_id)
        energy_path = os.path.join(path, energy_name)
        with open(energy_path, 'wb') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(("image_ix", "energy"))
            for image_id, energy in enumerate(energy_list):
                csv_writer.writerow((image_id, energy))


def get_r_at_k(base_dir, gt_map, n_images, file_suffix='_energy_values'):
    k_count = np.zeros(n_images)
    n_queries = len(gt_map)
    for i in range(0, n_queries):
        csv_name = 'q{:03d}{}.csv'.format(i, file_suffix)
        csv_path = os.path.join(base_dir, csv_name)
        if not os.path.isfile(csv_path):
            continue
        energies = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        sort_ix = np.argsort(energies[:, 1])
        recall = np.ones(n_images, dtype=np.float)
        for k in range(len(energies)):
            indices = np.array(energies[sort_ix][0:k + 1][:, 0])
            true_positives = set(indices) & set(gt_map[i])
            if len(true_positives) > 0:
                break
            recall[k] = 0.0
        k_count += recall
    return k_count / n_queries


def get_recall_values(data, ground_truth_map, n_images, show_plot=False,
                      save_path=None, do_holdout=False, x_limit=100):
    gen_plot = show_plot or save_path is not None
    if gen_plot:
        plot_handles = []
        plt.figure(1)
        plt.grid(True)

    all_values = []
    for path, label in data:
        vals = get_r_at_k(path, ground_truth_map, n_images)
        all_values.append(vals)
        if gen_plot:
            plot_handle = plt.plot(np.arange(len(vals)), vals, label=label)[0]
            plot_handles.append(plot_handle)

    if gen_plot:
        plt.xlabel('k')
        plt.ylabel('Recall at k')

        plt.legend(handles=plot_handles, loc=4)
        if x_limit > 0:
            plt.xlim([0, x_limit])
        plt.ylim([0, 1])

    if show_plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    return all_values


def get_false_negs(path):
    with open(path) as f:
        csv_reader = csv.reader(f)
        false_negs = [(int(query_index), int(image_index))
                      for query_index, image_index in list(csv_reader)[1:]]
    return false_negs


def get_text_parts(image_data, triple):
    sub = image_data.annotations.objects[triple.subject]
    obj = image_data.annotations.objects[triple.object]
    sub_name = np.array(sub.names).reshape(-1)[0]
    obj_name = np.array(obj.names).reshape(-1)[0]
    return (sub_name, triple.predicate, obj_name)


def get_indices(path, split):
    with open(os.path.join(path, '{}.txt'.format(split))) as f:
        return [int(line) for line in f.read().splitlines()]
