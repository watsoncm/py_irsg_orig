import os
import shutil
import json

import numpy as np
import scipy.io.matlab.mio5_params as siom
from tqdm import tqdm

import irsg_core.data_pull as dp
import irsg_core.image_fetch_utils as ifu
from image_query_data import ImageQueryData
from config import get_config_path

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']
    data_path = cfg_data['file_paths']['mat_path']


def generate_tp_neg(tp_data_pos, n_queries, n_images, negs_per_query):
    """Generate a series of negative examples for training the GMMs."""
    tp_data_neg = []
    for query_index in range(n_queries):
        image_indexes = [index for index in range(n_images)
                         if index not in tp_data_pos[query_index]]
        tp_data_neg.append(np.random.choice(image_indexes,
                                            size=negs_per_query))
    return tp_data_neg


def generate_all_query_plots(queries, if_data, condition_gmm=False,
                             visualize_gmm=False, negs_per_query=20,
                             use_geometric=False, suffix=None):
    """Generate plots for each given query."""
    tp_data_pos = ifu.get_partial_scene_matches(if_data.vg_data, queries)
    tp_data_neg = generate_tp_neg(tp_data_pos, len(queries),
                                  len(if_data.vg_data), negs_per_query)
    suffix_text = '' if suffix is None else '_' + suffix
    output_dir = os.path.join(out_path, 'query_viz' + suffix_text)
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
                                     compute_gt=is_pos,
                                     use_geometric=use_geometric)
                try:
                    iqd.compute_plot_data(condition_gmm=condition_gmm,
                                          visualize_gmm=visualize_gmm)
                except ValueError:
                    continue
                image_format = 'q{:03d}_i{:03d}_{}_{:.3f}.png'
                image_name = image_format.format(query_index, image_index,
                                                 label, iqd.model_total_pot)
                save_path = os.path.join(output_dir, image_name)
                iqd.generate_plot(save_path=save_path)
                energies[save_path] = iqd.model_total_pot

    # Copy over all relevant images
    top_pos_dir = os.path.join(out_path, 'query_top_pos' + suffix_text)
    top_neg_dir = os.path.join(out_path, 'query_top_neg' + suffix_text)
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
    """Generate a simple test plot to see if the program works."""
    iqd = ImageQueryData(queries['simple_graphs'][4], 4, 38, if_data)
    iqd.compute_plot_data(condition_gmm=True, visualize_gmm=True)
    iqd.generate_plot()


def gen_sro(sub, pred, obj, use_attrs=False):
    sub_struct, obj_struct = siom.mat_struct(), siom.mat_struct()
    sub_struct.__setattr__('names', sub)
    obj_struct.__setattr__('names', obj)

    rel_struct = siom.mat_struct()
    rel_struct.__setattr__('subject', 0)
    rel_struct.__setattr__('object', 1)
    rel_struct.__setattr__('predicate', pred)

    query_struct = siom.mat_struct()
    query_struct.__setattr__(
        'objects', np.array([sub_struct, obj_struct]))
    query_struct.__setattr__('unary_triples', np.array([]))
    query_struct.__setattr__('binary_triples', rel_struct)
    return query_struct


def _get_unary_triple(attr, is_sub):
    attr_struct = siom.mat_struct()
    attr_struct.__setattr__('subject', 0 if is_sub else 1)
    attr_struct.__setattr__('predicate', 'is')
    attr_struct.__setattr__('object', attr)
    return attr_struct


def gen_srao(sub, pred, attr, obj, use_attrs=False):
    query_struct = gen_sro(sub, pred, obj)
    obj_attr_struct = _get_unary_triple(attr, False)
    if use_attrs:
        query_struct.__setattr__('unary_triples', obj_attr_struct)
    return query_struct


def gen_asro(attr, sub, pred, obj, use_attrs=False):
    query_struct = gen_sro(sub, pred, obj)
    sub_attr_struct = _get_unary_triple(attr, True)
    if use_attrs:
        query_struct.__setattr__('unary_triples', sub_attr_struct)
    return query_struct


def gen_asrao(sub_attr, sub, pred, obj_attr, obj, use_attrs=False):
    query_struct = gen_sro(sub, pred, obj)
    if use_attrs:
        sub_attr_struct = _get_unary_triple(sub_attr, True)
        obj_attr_struct = _get_unary_triple(obj_attr, False)
        query_struct.__setattr__('unary_triples', np.array(
            [sub_attr_struct, obj_attr_struct]))
    return query_struct


def generate_queries_from_file(path, use_attrs=False):
    """Read queries from a specially-formatted file."""
    queries = []
    gen_dict = {'(sro)': gen_sro,
                '(srao)': gen_srao,
                '(asro)': gen_asro,
                '(asrao)': gen_asrao}
    with open(path) as f:
        for line in f.read().splitlines():
            query_struct = siom.mat_struct()
            parts = [part.replace('_', ' ') for part in line.split()]
            text_parts, gen_func = parts[:-1], gen_dict[parts[-1]]
            query_struct.annotations = gen_func(
                *text_parts, use_attrs=use_attrs)
            queries.append(query_struct)
    return queries


if __name__ == '__main__':
    _, _, _, _, queries, if_data = dp.get_all_data('psu', split='val',
                                                   use_csv=True)
    path = os.path.join(data_path, 'queries.txt')
    queries = generate_queries_from_file(path)
    generate_all_query_plots(queries, if_data, condition_gmm=True,
                             visualize_gmm=False)
    generate_all_query_plots(queries, if_data, condition_gmm=True,
                             visualize_gmm=False, use_geometric=True,
                             suffix='geom')
    # generate_test_plot(queries, if_data)
