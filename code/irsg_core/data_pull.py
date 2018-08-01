import json

import image_fetch_utils as ifu
import image_fetch_dataset as ifd
from config import get_config_path

# load data paths from the config file
cfg_file = open(get_config_path())
cfg_data = json.load(cfg_file)

out_path = cfg_data['file_paths']['output_path']
img_path = cfg_data['file_paths']['image_path']
mat_path = cfg_data['file_paths']['mat_path']
csv_path = cfg_data['file_paths']['csv_path']

# load model params
vgd = None
potentials = None
platt_mod = None
bin_mod = None
queries = None
ifdata = None
hf = None

data_loaded = {'test': False,
               'train': False}


def _load(use_csv=False, use_train=False):
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  dataset, other = ('train', 'test') if use_train else ('test', 'train')
  if not data_loaded[dataset]:
    data = ifu.get_mat_data(mat_path, get_potentials=(not use_csv))
    vgd, potentials, platt_mod, bin_mod, queries = data
    vg_string = 'vg_data_train' if use_train else 'vg_data_test'
    if use_csv:
      ifdata = ifd.CSVImageFetchDataset(vgd[vg_string], platt_mod, bin_mod, img_path, csv_path)
    else:
      ifdata = ifd.ImageFetchDataset(vgd[vg_string], potentials, platt_mod, bin_mod, img_path)
  data_loaded[dataset] = True
  data_loaded[other] = False

def get_all_data(use_csv=False, use_train=False):
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  _load(use_csv=use_csv, use_train=use_train)
  return vgd, potentials, platt_mod, bin_mod, queries, ifdata

def get_ifdata(use_csv=False, use_train=False):
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  _load(use_csv=use_csv, use_train=use_train)
  return ifdata

def get_supplemental_data(use_csv=False, use_train=False):
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  _load(use_csv=use_csv, use_train=use_train)
  return vgd, potentials, platt_mod, bin_mod, queries
