import csv
import collections
import json
import os
from datetime import datetime

import numpy as np
import scipy.io.matlab.mio5_params as siom
from tqdm import tqdm

import data_pull as dp
import image_fetch_core as ifc
import image_fetch_plot as ifp
import image_fetch_wrappers as ifw
import image_fetch_utils as ifu

with open('config.json') as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']


def image_batch_double(query, query_id, if_data, output_path, image_ix_subset):
    time_list = []
    energy_list = []
    image_format = '{}_q{:03d}i{:03d}.png'

    for i in image_ix_subset:
        inference_data = ifw.inference_pass(query, query_id, i, if_data, 'original')
        gm, tracker, energy, best_matches, marginals, duration = inference_data
        energy_list.append(energy)
        time_list.append(duration)

        energy_str = '{:06f}'.format(energy).replace('.', '_')
        image_name = image_format.format(energy_str, query_id, i)
        image_path = os.path.join(output_path, image_name)
        ifp.draw_best_objects(tracker, best_matches, energy, image_path)

    output_format = 'total time: {0}, average time: {1}'
    print(output_format.format(np.sum(time_list), np.mean(time_list)))

    csv_name = 'q{:03d}_energy_values.csv'.format(query_id)
    csv_path = os.path.join(output_path, csv_name)
    with open(csv_path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(("image_ix", "energy"))
        for i in range(len(image_ix_subset)):
            csv_writer.writerow((image_ix_subset[i], energy_list[i]))


def generate_query(subject, predicate):
    sub_struct = siom.mat_struct()
    sub_struct.names = subject

    rel_struct = siom.mat_struct()
    rel_struct.subject = 0
    rel_struct.object = 1
    rel_struct.predicate = predicate

    objects = np.array([sub_struct, sub_struct], dtype='O')
    anno_struct = siom.mat_struct()
    anno_struct.objects = objects
    anno_struct.unary_triples = np.array([])
    anno_struct.binary_triples = rel_struct

    query_struct = siom.mat_struct()
    query_struct.annotations = anno_struct
    return query_struct


# find top subjects and predicates 
vgd, _, platt_mod, bin_mod, queries, ifdata = dp.get_all_data(use_csv=True)
full_graphs = [q.annotations for q in vgd['vg_data_test']]
double_subjects = collections.Counter()
double_predicates = collections.defaultdict(list)
for query in full_graphs:
    for triple in query.binary_triples:
        subject, predicate, obj = triple.text
        if subject == obj:
            double_subjects[subject] += 1
            if predicate not in double_predicates[subject]:
                double_predicates[subject].append(predicate)

# create the new queries
new_queries = []
top_subjects = [subject for subject, _ in double_subjects.most_common(10)]
for subject in top_subjects:
    for predicate in double_predicates[subject]:
        new_queries.append(generate_query(subject, predicate))

# run all of the tests
tp_simple = ifu.get_partial_scene_matches(vgd['vg_data_test'], new_queries)
for index, query in enumerate(new_queries):
    now = datetime.now()
    subject = query.annotations.objects[0].names
    predicate = query.annotations.binary_triples.predicate.replace(' ', '_')
    scene_text = '{}_{}_{}'.format(subject, predicate, subject)

    batch_format = '{}_overlap_q{:03}_{}{:02}{:02}_{:02}{:02}{:02}/'
    batch_dir = batch_format.format(scene_text, index, now.year, now.month,
                                    now.day, now.hour, now.minute, now.second)
    batch_path = os.path.join(out_path, batch_dir)
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    image_batch_double(query.annotations, index, ifdata, batch_path,
                       tp_simple[index])
