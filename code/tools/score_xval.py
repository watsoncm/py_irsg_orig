import csv
import os
import json

import numpy as np
from tqdm import tqdm

import query_viz
import data_utils
import image_query_data
import irsg_core.data_pull as dp
import irsg_core.image_fetch_utils as ifu
from config import get_config_path
from image_query_data import ImageQueryData
from collections import defaultdict
from sklearn.metrics import roc_auc_score


WEIGHTS = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
K = 5


with open(get_config_path()) as f:
    cfg_data = json.load(f)
    data_path = cfg_data['file_paths']['mat_path']
    out_path = cfg_data['file_paths']['output_path']


def run_cross_val_test(if_data, queries, path, index, pred_weight):
    """Run a single test on the validation set."""
    image_query_data.generate_energy_data(queries, path, if_data,
                                          pred_weight=pred_weight)
    data_simple = [(path, 'xval {} energies'.format(index))]
    tp_simple = [list(sublist) for sublist in
                 ifu.get_partial_scene_matches(if_data.vg_data, queries)]
    recalls = data_utils.get_recall_values(data_simple, np.array(tp_simple),
                                           len(if_data.vg_data))[0]
    return recalls


def cross_validate_simple(if_data, queries, k_val, weight_values):
    """Do simple cross-validation over triples of IRSG weights."""
    query_format = 'query_energies_xval_{}/'
    result_format = 'pred_weight: {}\nR@{}: {}'
    pred_weights = np.array(weight_values)
    results = []
    recalls_at_k = []
    for index, pred_weight in tqdm(
            enumerate(pred_weights), total=pred_weights.size, desc='xval'):
        batch_path = os.path.join(out_path, query_format.format(index))
        if not os.path.exists(batch_path):
            os.mkdir(batch_path)
        recalls = run_cross_val_test(if_data, queries, batch_path,
                                     index, pred_weight)
        result = result_format.format(pred_weight, k_val, recalls[k_val])
        results.append(result)
        recalls_at_k.append(recalls[k_val])
        with open(os.path.join(batch_path, 'results.txt'), 'w') as f:
            f.write(result)

    with open(os.path.join(out_path, 'xval_results.txt'), 'w') as f:
        f.write(results[np.argmax(recalls_at_k)])


def cross_validate_tune_rcnns(if_data, k_val, iou_thresh=0.5):
    """Determine weights for each RCNN class given performance on the
    validation set."""
    # query_format = 'query_iou_recalls_{}/'
    gts = defaultdict(list)
    preds = defaultdict(list)
    for image_id in tqdm(range(len(if_data.vg_data)), desc='images'):
        if_data.configure(image_id, None, load_all=True)
        image_data = if_data.vg_data[image_id].annotations
        for name, values in tqdm(if_data.object_detections.iteritems(),
                                 total=len(if_data.object_detections),
                                 desc='dets'):
            gt_bboxes = []
            for obj in image_data.objects:
                if ImageQueryData.make_array(obj.names)[0] == name:
                    gt_bboxes.append(ImageQueryData.make_bbox(obj.bbox))
            for value in values:
                ious = [ImageQueryData.get_iou(value[:4], gt_bboxes)]
                gts[name].append(any([iou >= iou_thresh for iou in ious]))
                preds[name].append(value[4])

    auc_scores = {}
    for name in gts.keys():
        rcnn_gts = gts[name]
        rcnn_preds = preds[name]
        auc_scores[name] = roc_auc_score(rcnn_gts, rcnn_preds)

    import pdb; pdb.set_trace()
    with open(os.path.join(out_path, 'rcnn_tune_results.txt'), 'w') as f:
        csv_writer = csv.writer(f)
        for name, auc_score in auc_scores.iteritems():
            csv_writer.writerow((name, auc_score))


if __name__ == '__main__':
    if_data = dp.get_ifdata(dataset='psu', use_csv=True, split='val')
    query_path = os.path.join(data_path, 'queries.txt')
    queries = query_viz.generate_queries_from_file(query_path)
    cross_validate_simple(if_data, queries, K, WEIGHTS)
    # cross_validate_tune_rcnns(if_data, K)
