import json

import image_fetch_utils as ifu
import image_fetch_dataset as ifd
from config import get_config_path

with open(get_config_path()) as cfg_file:
    cfg_data = json.load(cfg_file)
    out_path = cfg_data['file_paths']['output_path']
    test_img_path = cfg_data['file_paths']['test_image_path']
    train_img_path = cfg_data['file_paths']['train_image_path']
    mat_path = cfg_data['file_paths']['mat_path']
    csv_path = cfg_data['file_paths']['csv_path']

vgd, potentials = None, None
platt_mod, bin_mod, queries = None, None, None
ifdata, hf = None, None
data_loaded = None


def _load(dataset='stanford', split='test', use_csv=False):
    """Helper function which loads data and caches it for later use."""
    global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
    if (dataset, split) != data_loaded:
        data = ifu.get_mat_data(mat_path, get_potentials=not use_csv,
                                get_bin_mod=not use_csv,
                                get_platt_mod=not use_csv)
        vgd, potentials, platt_mod, bin_mod, queries = data
        img_path = test_img_path if split == 'test' else train_img_path
        if use_csv:
            ifdata = ifd.CSVImageFetchDataset(
                dataset, split, vgd,  img_path, csv_path)
        else:
            if dataset != 'stanford':
                raise ValueError('Dataset {} invalid for loading without CSV'
                                 .format(dataset))

            # Note that for non-CSV, changing splits only changes vg_data
            if split == 'test':
                vg_dataset = vgd['vg_data_test']
            elif split == 'train':
                vg_dataset = vgd['vg_data_train']
            else:
                raise ValueError('Split {} invalid for stanford dataset'
                                 .format(split))
            ifdata = ifd.ImageFetchDataset(vg_dataset, potentials, platt_mod,
                                           bin_mod, img_path)
    data_loaded = (dataset, split)


def get_all_data(*args, **kwargs):
    """Retrieves an ImageFetchDataset object with all data inside, and
    all other auxillary data."""
    global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
    _load(*args, **kwargs)
    return vgd, potentials, platt_mod, bin_mod, queries, ifdata


def get_ifdata(*args, **kwargs):
    """Retrieves an ImageFetchDataset object with all data inside."""
    global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
    _load(*args, **kwargs)
    return ifdata


def get_supplemental_data(*args, **kwargs):
    """Retrieves all data which is not the ImageFetchDataset object."""
    global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
    _load(*args, **kwargs)
    return vgd, potentials, platt_mod, bin_mod, queries
