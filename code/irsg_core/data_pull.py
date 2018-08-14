import json

import image_fetch_utils as ifu
import image_fetch_dataset as ifd
from config import get_config_path

with open(get_config_path()) as cfg_file:
    cfg_data = json.load(cfg_file)
    out_path = cfg_data['file_paths']['output_path']
    img_path = cfg_data['file_paths']['image_path']
    mat_path = cfg_data['file_paths']['mat_path']
    csv_path = cfg_data['file_paths']['csv_path']

vgd, potentials = None, None
platt_mod, bin_mod, queries = None, None, None
ifdata, hf = None, None

data_loaded = {'train': False,
               'val': False,
               'test': False}


def _load(dataset='default', split='test', use_csv=False):
    global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
    if not data_loaded[split]:
        if dataset == 'default':
            get_potentials = not use_csv
            get_bin_mod = True
            get_platt_mod = True
        else:
            get_potentials = False
            get_bin_mod = False
            get_platt_mod = False
            if not use_csv:
                error = 'dataset {} must have use_csv=True'.format(dataset)
                raise ValueError(error)
        data = ifu.get_mat_data(mat_path, get_potentials=get_potentials,
                                get_bin_mod=get_bin_mod,
                                get_platt_mod=get_platt_mod)
        vgd, potentials, platt_mod, bin_mod, queries = data
        if use_csv:
            ifdata = ifd.CSVImageFetchDataset(dataset, split, vgd, platt_mod,
                                              bin_mod, img_path, csv_path)
        else:
            ifdata = ifd.ImageFetchDataset(split, vgd, potentials,
                                           platt_mod, bin_mod, img_path)

    for load_split in data_loaded.keys():
        data_loaded[load_split] = False
    data_loaded[dataset] = True


def get_all_data(*args, **kwargs):
    global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
    _load(*args, **kwargs)
    return vgd, potentials, platt_mod, bin_mod, queries, ifdata


def get_ifdata(*args, **kwargs):
    global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
    _load(*args, **kwargs)
    return ifdata


def get_supplemental_data(*args, **kwargs):
    global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
    _load(*args, **kwargs)
    return vgd, potentials, platt_mod, bin_mod, queries
