import json
import image_fetch_utils as ifu
import image_fetch_dataset as ifd



# load data paths from the config file
cfg_file = open('config.json')
cfg_data = json.load(cfg_file)

out_path = cfg_data['file_paths']['output_path']
img_path = cfg_data['file_paths']['image_path']
data_path = cfg_data['file_paths']['mat_path']

# load model params
vgd = None
potentials = None
platt_mod = None
bin_mod = None
queries = None
ifdata = None

data_loaded = False


def _load(use_hdf=False, reload=False):
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  if not data_loaded or reload:
    data = ifu.get_hdf_data(data_path) if use_hdf else ifu.get_mat_data(data_path)
    vgd, potentials, platt_mod, bin_mod, queries = data
    ifdata = ifd.ImageFetchDataset(vgd['vg_data_test'], potentials, platt_mod,
                                   bin_mod, img_path, use_hdf=use_hdf)
  data_loaded = True

def get_all_data(use_hdf=False, reload=False):
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  _load(use_hdf=use_hdf, reload=reload)
  return vgd, potentials, platt_mod, bin_mod, queries, ifdata

def get_ifdata(use_hdf=False, reload=False):
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  _load(use_hdf=use_hdf, reload=reload)
  return ifdata

def get_supplemental_data(use_hdf=False, reload=False):
  global data_loaded, vgd, potentials, platt_mod, bin_mod, queries, ifdata
  _load(use_hdf=use_hdf, reload=reload)
  return vgd, potentials, platt_mod, bin_mod, queries
