import os
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

from config import get_config_path

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']


def make_array(items_or_single):
    return np.array(items_or_single).reshape(-1)


def make_bbox(mat_bbox):
    return np.array([mat_bbox.x, mat_bbox.y, mat_bbox.w, mat_bbox.h])


def get_attributes(attr_triples, subject_id, object_id):
        sub_attrs, obj_attrs = [], []
        for attr_triple in attr_triples:
            if attr_triple.subject == subject_id:
                sub_attrs.append(attr_triple.object)
            elif attr_triple.subject == object_id:
                obj_attrs.append(attr_triple.object)
        return sub_attrs, obj_attrs


def get_text_parts(image_data, triple):
    sub = image_data.annotations.objects[triple.subject]
    obj = image_data.annotations.objects[triple.object]
    return make_array(sub.names)[0], triple.predicate, make_array(obj.names)[0]


def does_match(image, query):
    image_unary = make_array(image.annotations.unary_triples)
    image_binary = make_array(image.annotations.binary_triples)
    query_unary = make_array(query.annotations.unary_triples)
    query_binary = query.annotations.binary_triples

    query_sub, query_pred, query_obj = get_text_parts(query, query_binary)
    query_sub_attrs, query_obj_attrs = get_attributes(
        query_unary, query_binary.subject, query_binary.object)

    for image_triple in image_binary:
        image_sub, image_pred, image_obj = get_text_parts(image, image_triple)
        image_sub_attrs, image_obj_attrs = get_attributes(
            image_unary, image_triple.subject, image_triple.object)
        if (query_sub == image_sub and
                query_pred == image_pred and
                query_obj == image_obj and
                all([attr in image_sub_attrs for attr in query_sub_attrs]) and
                all([attr in image_obj_attrs for attr in query_obj_attrs])):
            return True
    return False


def get_partial_query_matches(images, queries):
    matches = []
    for query in queries:
        matches.append([image_id for image_id, image in enumerate(images)
                        if does_match(image, query)])
    return np.array(matches)


def get_r_at_k(base_dir, gt_map, n_images, file_suffix='_energy_values'):
    k_count = np.zeros(n_images)
    n_queries = len(gt_map)
    for i in range(0, n_queries):
        csv_name = 'q{:03d}{}.csv'.format(i, file_suffix)
        csv_path = os.path.join(base_dir, csv_name)
        if not os.path.isfile(csv_path):
            continue
        raw_energies = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        energies = np.atleast_2d(raw_energies)
        if energies.size == 0:
            continue
        sort_ix = np.argsort(energies[:, 1])
        recall = np.ones(n_images, dtype=np.float)
        for k in range(len(energies)):
            indices = np.array(energies[sort_ix][0:k + 1][:, 0])
            true_positives = set(indices) & set(gt_map[i])
            if len(true_positives) > 0:
                break
            recall[k] = 0.0
        k_count += recall
    return k_count / n_queries


def get_recall_values(data, ground_truth_map, n_images, show_plot=False,
                      save_path=None, do_holdout=False, x_limit=100):
    gen_plot = show_plot or save_path is not None
    if gen_plot:
        plot_handles = []
        plt.figure(1)
        plt.grid(True)

    all_values = []
    for path, label in data:
        vals = get_r_at_k(path, ground_truth_map, n_images)
        all_values.append(vals)
        if gen_plot:
            plot_handle = plt.plot(np.arange(len(vals)), vals, label=label)[0]
            plot_handles.append(plot_handle)

    if gen_plot:
        plt.xlabel('k')
        plt.ylabel('Recall at k')

        plt.legend(handles=plot_handles, loc=4)
        if x_limit > 0:
            plt.xlim([0, x_limit])
        plt.ylim([0, 1])

    if show_plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    return all_values


def get_iou_recall_values(data, ground_truth_map, n_images, show_plot=False,
                          save_path=None, do_holdout=False, y_limit=1.0):
    gen_plot = show_plot or save_path is not None
    if gen_plot:
        plot_handles = []
        plt.figure(1)
        plt.grid(True)

    all_values = []
    for base_dir, label in data:
        obj_ious = []
        for index in range(len(ground_truth_map)):
            csv_name = 'q{:03d}_iou_values.csv'.format(index)
            csv_path = os.path.join(base_dir, csv_name)
            if not os.path.isfile(csv_path):
                continue
            raw_csv_values = np.genfromtxt(
                csv_path, delimiter=',', skip_header=1)
            csv_values = np.atleast_2d(raw_csv_values)
            if csv_values.size == 0:
                continue
            obj_ious.extend(csv_values[:, 2])

        threshs = np.linspace(0.05, 0.5, num=2000)
        recalls = [sum([iou > thresh for iou in obj_ious]) /
                   float(len(obj_ious)) for thresh in threshs]
        all_values.append(zip(threshs, recalls))
        if gen_plot:
            plot_handle = plt.plot(threshs, recalls, label=label)[0]
            plot_handles.append(plot_handle)

    if gen_plot:
        plt.xlabel('IOU threshold')
        plt.ylabel('Object recall')
        plt.legend(handles=plot_handles, loc=1)
        plt.xlim([0.05, 0.5])
        plt.gca().invert_xaxis()
        if y_limit > 0:
            plt.ylim([0, y_limit])

    if show_plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    return all_values


def get_false_negs(path):
    false_negs = []
    with open(path) as f:
        csv_reader = csv.reader(f)
        for query_index, image_index in list(csv_reader)[1:]:
            false_negs.append((int(query_index), int(image_index)))
    return false_negs


def get_indices(path, split):
    with open(os.path.join(path, '{}.txt'.format(split))) as f:
        return [int(line) for line in f.read().splitlines()]


def save_platt_data(text, path, platt_a, platt_b):
    gmm_path = os.path.join(path, text)
    platt_path = os.path.join(gmm_path, 'platt.csv')
    with open(platt_path, 'w') as f:
        f.write('{},{}'.format(platt_a, platt_b))
