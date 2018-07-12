import json

import h5py

import image_fetch_utils as ifu
import image_fetch_dataset as ifd



# load data paths from the config file
cfg_file = open('config.json')
cfg_data = json.load(cfg_file)

out_path = cfg_data['file_paths']['output_path']
img_path = cfg_data['file_paths']['image_path']
mat_path = cfg_data['file_paths']['mat_path']
hdf_path = cfg_data['file_paths']['hdf_path']

# load model params
vgd = None
potentials = None
platt_mod = None
bin_mod = None
queries = None
ifdata = None
hf = None

data_loaded = False


def _load():
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  if not data_loaded:
    vgd, potentials, platt_mod, bin_mod, queries = ifu.get_mat_data(mat_path)
    ifdata = ifd.ImageFetchDataset(vgd['vg_data_test'], potentials, platt_mod, bin_mod, img_path)
  data_loaded = True

def get_all_data():
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  _load()
  return vgd, potentials, platt_mod, bin_mod, queries, ifdata

def get_ifdata():
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  _load()
  return ifdata

def get_supplemental_data():
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  _load()
  return vgd, potentials, platt_mod, bin_mod, queries

def get_hdf_data():
    global data_loaded, ifdata, hf
    if not data_loaded:
        hf = h5py.File(hdf_path, 'r')
        ifdata = ifd.HDFImageFetchDataset(hf, img_path)
    return ifdata

def close_hdf():
    global hf
    if hf is not None:
        hf.close()
        hf = None
