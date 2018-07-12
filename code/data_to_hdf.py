import csv
import os
import json

import h5py
import numpy as np
from tqdm import tqdm

import data_pull as dp

with open('config.json') as f:
    cfg_data = json.load(f)
    data_path = cfg_data['file_paths']['mat_path']


def transfer_annotations(base_group, annotations, is_simple_graph=False):
    # add triples
    unary_group = base_group.create_group('unary_triples')
    binary_group = base_group.create_group('binary_triples')
    unary_triples = annotations.unary_triples
    binary_triples = annotations.binary_triples
    str_dtype = h5py.special_dtype(vlen=str)

    # if we're transfering simple graph data, we don't have unary preds
    triple_iter = ((unary_group, unary_triples, 'O', not is_simple_graph),
                   (binary_group, binary_triples, int, True))
    for triple_group, triples, obj_dtype, use_predicate in triple_iter:
        triples = np.array(triples).reshape(-1)  # ensure 1D
        if use_predicate:
            predicates = np.array([triple.predicate for triple in triples],
                                   dtype='O')
        subjects = np.array([triple.subject for triple in triples])
        objects = np.array([triple.object for triple in triples],
                            dtype=obj_dtype)

        # get the right dtype and place in HDF
        obj_data_dtype = str_dtype if obj_dtype is 'O' else obj_dtype
        if use_predicate:
            triple_group.create_dataset('predicates', data=predicates,
                                        dtype=str_dtype)
        triple_group.create_dataset('subjects', data=subjects)
        triple_group.create_dataset('objects', data=objects,
                                    dtype=obj_data_dtype)

    # add objects
    image_objs = annotations.objects
    objects_group = base_group.create_group('objects')
    for obj_idx, obj in enumerate(image_objs):
        object_name = 'object_{}'.format(obj_idx)
        object_group = objects_group.create_group(object_name)
        names = np.array(obj.names, dtype='O').reshape(-1)  # ensure 1D
        object_group.create_dataset('names', data=obj.names,
                                    dtype=str_dtype)

        # simple graphs don't provide bounding boxes
        if not is_simple_graph:
            bbox = np.array([obj.bbox.x, obj.bbox.y,
                             obj.bbox.w, obj.bbox.h])
            object_group.attrs['bbox'] = bbox


def transfer_vgd(hf, vgd):
    """Transfer VGD data into a HDF5 store."""
    vgd_group = hf.create_group('vgd')
    for group_name, desc in (('vg_data_test', 'vgd test'),
                             ('vg_data_train', 'vgd train')):
        dataset_group = vgd_group.create_group(group_name)
        vgd_data = vgd[group_name]
        for img_idx in tqdm(range(vgd_data.shape[0]), desc=desc):
            # add image url
            image_name = 'image_{}'.format(img_idx)
            image_group = dataset_group.create_group(image_name)
            image_group.attrs['image_url'] = vgd_data[img_idx].image_url
            transfer_annotations(image_group, vgd_data[img_idx].annotations)


def transfer_potentials(hf, potentials):
    """Transfer potential data into a HDF5 store."""
    pot_data = potentials['potentials_s']
    class_to_idx = pot_data.class_to_idx.serialization
    pot_group = hf.create_group('potentials')

    # add boxes and scores
    box_group = pot_group.create_group('boxes')
    score_group = pot_group.create_group('scores')
    pairs = enumerate(zip(pot_data.boxes, pot_data.scores))
    for box_idx, (box, scores) in tqdm(pairs, desc='boxes/scores',
                                       total=pot_data.boxes.shape[0]):
        image_name = 'image_{}'.format(box_idx)
        box_group.create_dataset(image_name, data=box)
        score_group.create_dataset(image_name, data=scores)

    # add class names
    str_dtype = h5py.special_dtype(vlen=str)
    pot_group.create_dataset('classes', data=pot_data.classes, dtype=str_dtype)

    # add class to index data
    class_group = pot_group.create_group('class_to_idx')
    class_group.create_dataset('keys', data=class_to_idx.keys, dtype=str_dtype)
    class_group.create_dataset('values', data=class_to_idx.values.astype(int))


def transfer_platt_mod(hf, platt_mod):
    """Transfer Platt scaling models into a HDF5 store."""
    platt_data = platt_mod['platt_models'].s_models.serialization
    platt_group = hf.create_group('platt_mod')

    # transfer unary platt model data
    s_group = platt_group.create_group('s_models')
    str_dtype = h5py.special_dtype(vlen=str)
    s_group.create_dataset('keys', data=platt_data.keys, dtype=str_dtype)
    s_group.create_dataset('values', data=np.array(list(platt_data.values)))


def transfer_bin_mod(hf, bin_mod):
    """Transfer GMM parameters into a HDF5 store."""
    bin_mod_group = hf.create_group('bin_mod')
    for mod_name in bin_mod.keys():
        mod_group = bin_mod_group.create_group(mod_name)
        mod_data = bin_mod[mod_name]
        platt_data = np.array([mod_data.platt_a, mod_data.platt_b])
        mod_group.create_dataset('platt_params', data=platt_data)
        mod_group.create_dataset('gmm_weights', data=mod_data.gmm_weights)
        mod_group.create_dataset('gmm_mu', data=mod_data.gmm_mu)
        mod_group.create_dataset('gmm_sigma', data=mod_data.gmm_sigma)


def transfer_queries(hf, queries):
    """Transfer simple graph queries into a HDF5 store."""
    queries_group = hf.create_group('queries')
    simple_group = queries_group.create_group('simple_graphs')
    query_data = queries['simple_graphs']
    for query_idx in tqdm(range(query_data.shape[0]), desc='graphs'):
        query_name = 'query_{}'.format(query_idx)
        query_group = simple_group.create_group(query_name)
        transfer_annotations(query_group, query_data[query_idx].annotations,
                             is_simple_graph=True)


def convert_all_to_hdf():
    """Converts all Matlab files into a single HDF file."""
    vgd, potentials, platt_mod, bin_mod, queries = dp.get_supplemental_data()
    path = os.path.join(data_path, 'all_data.h5')
    with h5py.File(path, 'w') as hf:
        transfer_vgd(hf, vgd)
        transfer_potentials(hf, potentials)
        transfer_platt_mod(hf, platt_mod)
        transfer_bin_mod(hf, bin_mod)
        transfer_queries(hf, queries)


if __name__ == '__main__':
    convert_all_to_hdf()
