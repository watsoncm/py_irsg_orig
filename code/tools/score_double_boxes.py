import csv
import collections
import json
import os
from datetime import datetime

import numpy as np
import scipy.io.matlab.mio5_params as siom
from tqdm import tqdm

import irsg_core.data_pull as dp
import irsg_core.image_fetch_core as ifc
import irsg_core.image_fetch_plot as ifp
import irsg_core.image_fetch_wrappers as ifw
import irsg_core.image_fetch_utils as ifu
from gmm_viz import visualize_gmm

CSV_DESCRIPTION = '''# query_index: which constructed query we're on
# image_index: which image we're on
# subject: subject of the constructed query (also the object)
# predicate: relationship between the subject and object
# binary_score: binary score before platt scaling of boxes model chose
# mean_gt_binary_score: mean score of all ground truth pairs in image
# binary_model_used: model used to calculate binary scores
# platt_a: the first Platt parameter
# platt_b: the second Platt parameter
# double_boxed: whether or not the model chose the same box twice

'''

with open('config.json') as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']


def image_batch_double(query, query_id, if_data, output_path, image_ix_subset):
    time_list = []
    energy_list = []
    image_format = '{}_q{:03d}i{:03d}.png'

    for i in image_ix_subset:
        inference_data = ifw.inference_pass(query, query_id, i,
                                            if_data, 'original')
        _, tracker, energy, best_matches, _, duration = inference_data
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


def gmm_plot_double(query, query_id, if_data, output_path, image_ix_subset):
    _, model_name = get_relationship_model(query.annotations,
                                           ifdata.relationship_models)
    if model_name is not None:
        # get the query's subject, predicate, and object
        query_objs = query.annotations.objects
        query_sub_idx = query.annotations.binary_triples.subject
        query_sub = np.array(query_objs[query_sub_idx].names).reshape(-1)[0]
        query_pred = query.annotations.binary_triples.predicate

        # make gmm plots for each image
        for image_index in image_ix_subset:
            anno = ifdata.vg_data[image_index].annotations
            match = None
            for triple in anno.binary_triples:

                # get the triple's subject, predicate, and object
                # note: we use the text attribute, because some objects
                # aren't linked properly (see image 603, man/jacket)
                trip_sub, trip_pred, trip_obj = triple.text

                # if it's a match to our query, use it
                if (trip_sub == query_sub
                        and trip_obj == query_sub
                        and trip_pred == query_pred):
                    match = triple

            # create gmm plots
            inference_data = ifw.inference_pass(query.annotations, query_id,
                                                image_index, if_data,
                                                'original')
            _, _, energy, best_matches, _, duration = inference_data
            image_name = 'q{:03d}_i{:03d}.png'.format(query_id, image_index)
            save_path = os.path.join(output_path, image_name)
            visualize_gmm(model_name, image_index, match.subject, match.object,
                          if_data.vg_data, if_data.relationship_models, ifdata,
                          save_path=save_path)


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


def get_double_queries(vgd):
    # find top subjects and predicates
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
    top_subjects = [sub for sub, _ in
                    double_subjects.most_common(10)]
    for subject in top_subjects:
        for predicate in double_predicates[subject]:
            new_queries.append(generate_query(subject, predicate))
    return new_queries


def draw_double_plots(queries, tp_data, ifdata, gmm_plot=True):
    for index, query in enumerate(queries):
        # set up batch directory
        now = datetime.now()
        subject = query.annotations.objects[0].names
        triples = query.annotations.binary_triples
        predicate = triples.predicate.replace(' ', '_')
        scene_text = '{}_{}_{}'.format(subject, predicate, subject)
        batch_format = '{}_overlap_q{:03}_{}{:02}{:02}_{:02}{:02}{:02}/'
        batch_dir = batch_format.format(scene_text, index, now.year, now.month,
                                        now.day, now.hour, now.minute,
                                        now.second)
        batch_path = os.path.join(out_path, batch_dir)
        os.mkdir(batch_path)

        if gmm_plot:
            gmm_plot_double(query, index, ifdata, batch_path, tp_data[index])
        else:
            image_batch_double(query, index, ifdata, batch_path,
                               tp_data[index])


def score_boxes(subject_box, object_box, rel_params):
    sub_center = (subject_box[0] + 0.5 * subject_box[2],
                  subject_box[1] + 0.5 * subject_box[3])
    obj_center = (object_box[0] + 0.5 * object_box[2],
                  object_box[1] + 0.5 * object_box[3])
    rel_center = ((sub_center[0] - obj_center[0]) / subject_box[2],
                  (sub_center[1] - obj_center[1]) / subject_box[3])
    rel_dims = (object_box[2] / subject_box[2],
                object_box[3] / subject_box[3])
    features = np.vstack((rel_center[0], rel_center[1],
                          rel_dims[1], rel_dims[0])).T

    scores = ifc.gmm_pdf(features, rel_params.gmm_weights,
                         rel_params.gmm_mu, rel_params.gmm_sigma)
    eps = np.finfo(np.float).eps
    scores = np.log(eps + scores)
    sig_scores = 1.0 / (1.0 + np.exp(rel_params.platt_a * scores +
                                     rel_params.platt_b))

    return scores[0], sig_scores[0]


def get_relationship_model(query, rel_mod):
    subject = query.objects[0].names
    rel = query.binary_triples.predicate
    clean_rel = rel.replace(' ', '_')
    bin_trip_key = '{}_{}_{}'.format(subject, clean_rel, subject)
    if bin_trip_key not in rel_mod:
        bin_trip_key = '*_{}_*'.format(clean_rel)
    try:
        return rel_mod[bin_trip_key], bin_trip_key
    except KeyError:
        return None, None


def quiet_inference_pass(query, query_id, image_ix, if_data):
    if_data.configure(image_ix, query)
    gm, tracker = ifc.generate_pgm(if_data, verbose=False)
    energy, indices, marginals = ifc.do_inference(gm)
    return gm, tracker, energy, indices, marginals


def test_gmm_density(queries, tp_data, vgd, bin_mod, ifdata, output_path):
    with open(output_path, 'wb') as f:
        f.write(CSV_DESCRIPTION)
        csv_writer = csv.writer(f, lineterminator='\n')
        csv_writer.writerow(('query_index', 'image_index', 'subject',
                             'predicate', 'binary_score',
                             'max_gt_binary_score',
                             'mean_gt_binary_score', 'binary_prob',
                             'mean_gt_binary_prob', 'binary_model_used',
                             'platt_a', 'platt_b', 'double_boxed'))
        for query_index, query in tqdm(enumerate(queries), total=len(queries),
                                       desc='double queries'):
            rel_params, rel_name = get_relationship_model(query.annotations,
                                                          bin_mod)
            if rel_params is None:
                continue
            for image_index in tp_data[query_index]:
                # gather important data
                subject = query.annotations.objects[0].names
                rel = query.annotations.binary_triples.predicate

                # run inference
                pass_data = quiet_inference_pass(query.annotations,
                                                 query_index,
                                                 image_index, ifdata)
                _, tracker, _, best_matches, _ = pass_data
                boxes = [tracker.box_pairs[index][1][:4]
                         for index in best_matches]
                annotations = vgd['vg_data_test'][image_index].annotations

                # get all ground truth boxes
                gt_box_pairs = []
                for triple in annotations.binary_triples:
                    if np.array_equal(triple.text, [subject, rel, subject]):
                        subject_box_obj = annotations.objects[triple.subject].bbox
                        subject_box = np.array([subject_box_obj.x, subject_box_obj.y,
                                                subject_box_obj.w, subject_box_obj.h])
                        object_box_obj = annotations.objects[triple.object].bbox
                        object_box = np.array([object_box_obj.x, object_box_obj.y,
                                               object_box_obj.w, object_box_obj.h])
                        gt_box_pairs.append((subject_box, object_box))

                # write final scores
                box_scores = score_boxes(*boxes, rel_params=rel_params)
                gt_box_scores = [score_boxes(*pair, rel_params=rel_params)
                                 for pair in gt_box_pairs]
                gt_box_firsts = [pair[0] for pair in gt_box_scores]
                gt_box_seconds = [pair[1] for pair in gt_box_scores]
                double_boxed = np.array_equal(boxes[0], boxes[1])
                csv_writer.writerow((query_index, image_index,
                                     subject, rel,
                                     box_scores[0], np.max(gt_box_firsts),
                                     np.mean(gt_box_firsts),
                                     box_scores[1], np.mean(gt_box_seconds),
                                     rel_name, rel_params.platt_a,
                                     rel_params.platt_b, double_boxed))


if __name__ == '__main__':
    vgd, _, _, bin_mod, _, ifdata = dp.get_all_data(use_csv=True)
    queries = get_double_queries(vgd)
    tp_simple = ifu.get_partial_scene_matches(vgd['vg_data_test'], queries)
    draw_double_plots(queries, tp_simple, ifdata)
    output_path = os.path.join(out_path, 'gmm_density.csv')
    test_gmm_density(queries, tp_simple, vgd, bin_mod, ifdata, output_path)
