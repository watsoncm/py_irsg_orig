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


def transfer_vgd(hf, vgd):
    """Transfer VGD data into a HDF5 store."""
    vgd_group = hf.create_group('vgd')
    for group_name, desc in (('vg_data_test', 'vgd test'),
                             ('vg_data_train', 'vgd train')):
        dataset_group = vgd_group.create_group(group_name)
        data = vgd[group_name]
        for img in tqdm(range(data.shape[0]), desc=desc):
            # add image url
            image_name = 'image_{}'.format(img)
            image_group = dataset_group.create_group(image_name)
            image_group.attrs['image_url'] = data[img].image_url

            # add triples
            unary_group = image_group.create_group('unary_triples')
            binary_group = image_group.create_group('binary_triples')
            unary_triples = data[img].annotations.unary_triples
            binary_triples = data[img].annotations.binary_triples
            str_dtype = h5py.special_dtype(vlen=str)

            for triple_group, triples in ((unary_group, unary_triples),
                                          (binary_group, binary_triples)):
                predicates = np.array([triple.predicate for triple in triples],
                                      dtype='O')
                subjects = np.array([triple.subject for triple in triples])
                objects = np.array([triple.object for triple in triples])
                triple_group.create_dataset('predicates', data=predicates,
                                            dtype=str_dtype)
                triple_group.create_dataset('subjects', data=subjects)
                triple_group.create_dataset('objects', data=objects)

            # add objects
            image_objs = data[img].annotations.objects
            objects_group = image_group.create_group('objects')
            for obj_idx, obj in enumerate(image_objs):
                object_name = 'object_{}'.format(obj_idx)
                object_group = objects_group.create_group(object_name)
                names = np.array(obj.names, dtype='O')
                object_group.create_dataset('names', data=obj.names,
                                            dtype=str_dtype)
                bbox = np.array([obj.bbox.x, obj.bbox.y,
                                 obj.bbox.w, obj.bbox.h])
                object_group.attrs['bbox'] = bbox


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
        box_name = 'box_{}'.format(box_idx)
        box_group.create_dataset(box_name, data=box)
        score_group.create_dataset(box_name, data=scores)

    # add class to index data
    class_group = pot_group.create_group('class_to_idx')
    str_dtype = h5py.special_dtype(vlen=str)
    class_group.create_dataset('keys', data=class_to_idx.keys, dtype=str_dtype)
    class_group.create_dataset('values', data=class_to_idx.values.astype(int))


def convert_all_to_hdf():
    """Converts all Matlab files into a single HDF file."""
    vgd, potentials, platt_mod, bin_mod, queries = dp.get_supplemental_data()
    path = os.path.join(data_path, 'all_data.h5')
    with h5py.File(path, 'w') as hf:
        transfer_vgd(hf, vgd)
        import pdb; pdb.set_trace()
        transfer_potentials(hf, potentials)

if __name__ == '__main__':
    convert_all_to_hdf()
