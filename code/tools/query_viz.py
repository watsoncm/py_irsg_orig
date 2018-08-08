import os
import shutil
import json

import numpy as np
import scipy.io.matlab.mio5_params as siom
from tqdm import tqdm

import irsg_core.data_pull as dp
import irsg_core.image_fetch_utils as ifu
import irsg_core.image_fetch_querygen as ifq
from image_query_data import ImageQueryData
from config import get_config_path

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']
    data_path = cfg_data['file_paths']['mat_path']


def generate_tp_neg(tp_data_pos, n_queries, n_images, negs_per_query):
    tp_data_neg = []
    for query_index in range(n_queries):
        image_indexes = [index for index in range(n_images)
                         if index not in tp_data_pos[query_index]]
        tp_data_neg.append(np.random.choice(image_indexes,
                                            size=negs_per_query))
    return tp_data_neg


def generate_all_query_plots(queries, if_data, condition_gmm=False,
                             visualize_gmm=False, negs_per_query=20):
    tp_data_pos = ifu.get_partial_scene_matches(if_data.vg_data, queries)
    tp_data_neg = generate_tp_neg(tp_data_pos, len(queries),
                                  len(if_data.vg_data), negs_per_query)
    output_dir = os.path.join(out_path, 'query_viz')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    positive_energies = {}
    negative_energies = {}

    # Run all positives and negatives
    for tp_data, energies, is_pos in ((tp_data_pos, positive_energies, True),
                                      (tp_data_neg, negative_energies, False)):
        label = 'pos' if is_pos else 'neg'
        for query_index, query in tqdm(enumerate(queries), desc='graphs',
                                       total=len(queries)):
            for image_index in tp_data[query_index]:
                iqd = ImageQueryData(query, query_index, image_index, if_data,
                                     compute_gt=is_pos)
                try:
                    iqd.compute_plot_data(condition_gmm=condition_gmm,
                                          visualize_gmm=visualize_gmm)
                except ValueError:
                    continue
                image_format = 'q{:03d}_i{:03d}_{}_{:.2f}.png'
                image_name = image_format.format(query_index, image_index,
                                                 label, iqd.model_energy)
                save_path = os.path.join(output_dir, image_name)
                iqd.generate_plot(save_path=save_path)
                energies[save_path] = iqd.model_energy

    # Copy over all relevant images
    top_pos_dir = os.path.join(out_path, 'query_top_pos')
    top_neg_dir = os.path.join(out_path, 'query_top_neg')
    if not os.path.exists(top_pos_dir):
        os.mkdir(top_pos_dir)
    if not os.path.exists(top_neg_dir):
        os.mkdir(top_neg_dir)
    pos_pairs = sorted(positive_energies.items(), key=lambda pair: pair[1])
    neg_pairs = sorted(negative_energies.items(), key=lambda pair: pair[1])
    for path, _ in pos_pairs[:10]:
        shutil.copy(path, top_pos_dir)
    for path, _ in neg_pairs[:10]:
        shutil.copy(path, top_neg_dir)


def generate_test_plot(queries, if_data):
    iqd = ImageQueryData(queries['simple_graphs'][4], 4, 38, if_data)
    iqd.compute_plot_data(condition_gmm=True, visualize_gmm=True)
    iqd.generate_plot()


def generate_queries_from_file(path):
    queries = []
    gen_dict = {'(sro)': ifq.gen_sro,
                '(srao)': ifq.gen_srao,
                '(asro)': ifq.gen_asro,
                '(asrao)': ifq.gen_asrao}
    with open(path) as f:
        for line in f.read().splitlines():
            query_struct = siom.mat_struct()
            parts = line.split()
            text, gen_func = ' '.join(parts[:-1]), gen_dict[parts[-1]]
            query_struct.annotations = gen_func(text)
            queries.append(query_struct)
    return queries


if __name__ == '__main__':
    _, _, _, _, queries, if_data = dp.get_all_data(use_csv=True)
    path = os.path.join(data_path, 'queries.txt')
    queries = generate_queries_from_file(path)
    generate_all_query_plots(queries, if_data, condition_gmm=True,
                             visualize_gmm=False)
    # generate_test_plot(queries, if_data)
