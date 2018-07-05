import json
import numpy as np

import data_pull as dp

with open('config.json') as f:
    cfg_data = json.load(f)
    csv_path = cfg_data['file_paths']['csv_path']


def convert_all_to_csv():
    """Converts all Matlab files into corresponding CSV files."""
    vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
    run_path = os.path.join(csv_path, 'run_files')
    rel_path = os.path.join(csv_path, 'rel_files')

    for path in (run_path, rel_path):
        if not os.path.exists(path):
            os.mkdir(path)

    # run_path
    for image_idx in np.arange(ifdata.vg_data.size):
        ifdata.configure(image_idx, None)
        csv_name = 'irsg_{}.csv'.format(image_idx)
        for obj_name, detections in ifdata.object_detections:
            obj_name = obj_name[4:]  # remove "obj:" prefix
            obj_path = os.path.join(run_path, obj_name)
            if not os.path.exists(obj_path):
                os.mkdir(obj_path)


