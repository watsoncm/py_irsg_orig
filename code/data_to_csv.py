import os
import json
import pickle

import numpy as np
from tqdm import tqdm

import data_pull as dp
import image_fetch_core as ifc

with open('config.json') as f:
    cfg_data = json.load(f)
    csv_path = cfg_data['file_paths']['csv_path']


def write_obj_attr_scores(out_path, detections):
    """Write object/attribute detection scores to CSV."""
    neg_col = -np.ones((detections.shape[0], 1))  # for compatibility
    detections_full = np.hstack((detections, neg_col))
    np.savetxt(out_path, detections_full,
               fmt='%i,%i,%i,%i,%.6f,%i')


def transfer_obj_attr_scores(ifdata, root_path, desc, use_attrs=False,
                             progress=True):
    """Transfer scores from Matlab format to CSV."""
    # TODO: remove [:10]
    for image_idx in tqdm(np.arange(ifdata.vg_data.size)[:10], desc=desc):
        ifdata.configure(image_idx, None)
        csv_name = 'irsg_{}.csv'.format(image_idx)
        detection_dict = (ifdata.attribute_detections if use_attrs else
                          ifdata.object_detections)
        for name, detections in detection_dict.iteritems():
            name = name[4:]
            path = os.path.join(root_path, name)
            if not os.path.exists(path):
                os.mkdir(path)
            img_csv_path = os.path.join(path, csv_name)
            write_obj_attr_scores(img_csv_path, detections)


def write_rel_scores(out_path, params):
    """Write relationship scores to CSV."""
    # _, tracker = generate_pgm(if_data


def transfer_rel_scores(ifdata, root_path, desc, progress=True):
    for image_idx in tqdm(np.arange(ifdata.vg_data.size), desc=desc):
        ifdata.configure(image_idx, None)
        csv_name = 'irsg_{}.csv'.format(image_idx)
        _, tracker = ifc.generate_pgm(ifdata)
        for relationship in tracker.relationships:
            pass
            # log.log_likelihood


def convert_all_to_csv(progress=True):
    """Converts all Matlab files into corresponding CSV files."""
    ifdata = dp.get_ifdata()
    obj_path = os.path.join(csv_path, 'obj_files')
    attr_path = os.path.join(csv_path, 'attr_files')
    rel_path = os.path.join(csv_path, 'rel_files')

    for path in (obj_path, attr_path, rel_path):
        if not os.path.exists(path):
            os.mkdir(path)

    transfer_obj_attr_scores(ifdata, obj_path, desc='objects',
                             progress=progress)
    transfer_obj_attr_scores(ifdata, attr_path, desc='attributes',
                             use_attrs=False, progress=progress)
    transfer_rel_scores(ifdata, rel_path, desc='relationships',
                        progress=progress)


if __name__ == '__main__':
    convert_all_to_csv()
