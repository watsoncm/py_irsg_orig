import os
import json
import pickle
import itertools

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


def write_rel_scores(out_path, ifdata, params):
    """Write relationship scores to CSV."""
    boxes = ifdata.object_detections.values()[0]
    print('size:')
    print(boxes.size)

    print('first val')
    print(boxes[0][0])
    cart_prod = np.array(list(itertools.product(boxes, repeat=2)))

    sub_centerx = cart_prod[:, 0, 0] + 0.5 * cart_prod[:, 0, 2]
    sub_centery = cart_prod[:, 0, 1] + 0.5 * cart_prod[:, 0, 3]
    obj_centerx = cart_prod[:, 1, 1] + 0.5 * cart_prod[:, 1, 3]
    obj_centery = cart_prod[:, 1, 1] + 0.5 * cart_prod[:, 1, 3]

    rel_centerx = (sub_centerx - obj_centerx) / cart_prod[:, 0, 2]
    rel_centery = (sub_centery - obj_centery) / cart_prod[:, 0, 3]
    rel_height = cart_prod[:, 1, 2] / cart_prod[:, 0, 2]
    rel_width = cart_prod[:, 1, 3] / cart_prod[:, 0, 3]
    features = np.vstack((rel_centerx, rel_centery, rel_height, rel_width)).T

    raw_scores = ifc.gmm_pdf(features, params.gmm_weights, params.gmm_mu,
                             params.gmm_sigma)
    log_scores = np.log(raw_scores + np.finfo(np.float).eps)
    platt_scores = params.platt_a * log_scores + params.platt_b
    scores = 1.0 / (1.0 + np.exp(platt_scores))
    np.savetxt(out_path, scores.reshape((boxes.shape[0], boxes.shape[0])),
               delimiter=',')


def transfer_scores(ifdata, root_path, score_type='obj', progress=True):
    """Transfer scores from Matlab format to CSV."""
    if score_type == 'obj':
        desc = 'objects'
    elif score_type == 'attr':
        desc = 'attributes'
    elif score_type == 'rel':
        desc = 'relationships'

    # TODO: remove [:10]
    for image_idx in tqdm(np.arange(ifdata.vg_data.size)[:10], desc=desc):
        ifdata.configure(image_idx, None)

        if score_type == 'obj':
            detection_dict = ifdata.object_detections
        elif score_type == 'attr':
            detection_dict = ifdata.attribute_detections
        elif score_type == 'rel':
            detection_dict = ifdata.relationship_models

        csv_name = 'irsg_{}.csv'.format(image_idx)
        for name, result in tqdm(detection_dict.iteritems(),
                                 total=len(detection_dict)):
            if score_type != 'rel':
                name = name[4:]
            path = os.path.join(root_path, name)
            if not os.path.exists(path):
                os.mkdir(path)
            img_csv_path = os.path.join(path, csv_name)
            if score_type != 'rel':
                write_obj_attr_scores(img_csv_path, result)
            else:
                write_rel_scores(img_csv_path, ifdata, result)


def convert_all_to_csv(progress=True):
    """Converts all Matlab files into corresponding CSV files."""
    ifdata = dp.get_ifdata()
    obj_path = os.path.join(csv_path, 'obj_files')
    attr_path = os.path.join(csv_path, 'attr_files')
    rel_path = os.path.join(csv_path, 'rel_files')

    for path in (obj_path, attr_path, rel_path):
        if not os.path.exists(path):
            os.mkdir(path)

    transfer_scores(ifdata, obj_path, score_type='obj', progress=progress)
    transfer_scores(ifdata, attr_path, score_type='attr', progress=progress)
    transfer_scores(ifdata, rel_path, score_type='rel', progress=progress)


if __name__ == '__main__':
    convert_all_to_csv()
