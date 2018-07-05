import json
import numpy as np

import data_pull as dp

with open('config.json') as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']

vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
for image_idx in np.arange(vgd['vg_data_test'].size): 
    ifdata.configure(image_idx, None)
    print(ifdata.object_detections.keys())
    print(ifdata.attribute_detections)
