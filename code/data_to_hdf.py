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


def convert_all_to_hdf():
    """Converts all Matlab files into a single HDF file."""
    vgd, potentials, platt_mod, bin_mod, queries = dp.get_supplemental_data()
    pot_data = potentials['potentials_s']
    class_to_idx = pot_data.class_to_idx.serialization
    path = os.path.join(data_path, 'all_data.h5')
    with h5py.File(path, 'w') as hf:
        pot_group = hf.create_group('potentials')
        box_group = pot_group.create_group('boxes')
        score_group = pot_group.create_group('scores')
        pairs = enumerate(zip(pot_data.boxes, pot_data.scores))
        for box_idx, (box, scores) in tqdm(pairs, desc='boxes/scores',
                                           total=pot_data.boxes.shape[0]):
            box_name = 'box_{}'.format(box_idx)
            box_group.create_dataset(box_name, data=box)
            score_group.create_dataset(box_name, data=scores)

        class_group = pot_group.create_group('class_to_idx')
        str_dtype = h5py.special_dtype(vlen=str)
        class_group.create_dataset('keys', data=class_to_idx.keys, dtype=str_dtype)
        import pdb; pdb.set_trace()
        class_group.create_dataset('values', data=class_to_idx.values.astype(int))


if __name__ == '__main__':
    convert_all_to_hdf()
