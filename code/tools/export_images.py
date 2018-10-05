import os
import shutil
import json

import query_viz
import data_utils
import irsg_core.data_pull as dp
from image_query_data import ImageQueryData
from config import get_config_path
from tqdm import tqdm

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    data_path = cfg_data['file_paths']['mat_path']
    out_path = cfg_data['file_paths']['output_path']
    sg_path = cfg_data['file_paths']['sg_path']


def export_images(queries, if_data, false_negs, path, json_data):
    """Save the counts of the given queries to a file."""
    tp_simple = data_utils.get_partial_query_matches(
        if_data.vg_data, queries)
    for query_index, image_index in false_negs:
        tp_simple[query_index].append(image_index)

    pairs = enumerate(zip(queries, tp_simple))
    for index, (query, image_indexes) in tqdm(
            pairs, desc='queries', total=len(queries)):
        iqd = ImageQueryData(query, index, 0, if_data)
        query_dir = iqd.get_query_text().replace(' ', '_')
        query_path = os.path.join(path, query_dir)
        if not os.path.exists(query_path):
            os.makedirs(query_path)
        for image_index in image_indexes:
            if_data.configure(image_index, None)
            shutil.copy(if_data.image_filename, query_path)
            image_name = os.path.basename(if_data.image_filename)
            json_name = os.path.splitext(image_name)[0] + '.json'
            with open(os.path.join(query_path, json_name), 'w') as f:
                json.dump(json_data[image_index], f)

    neg_path = os.path.join(path, 'negative_images')
    if not os.path.exists(neg_path):
        os.makedirs(neg_path)
    full_negs = [index for index in range(len(if_data.vg_data))
                 if all([index not in image_index
                         for image_index in tp_simple])]

    for image_index in full_negs:
        if_data.configure(image_index, None)
        shutil.copy(if_data.image_filename, neg_path)
        image_name = os.path.basename(if_data.image_filename)
        json_name = os.path.splitext(image_name)[0] + '.json'
        with open(os.path.join(neg_path, json_name), 'w') as f:
            json.dump(json_data[image_index], f)


if __name__ == '__main__':
    if_data = dp.get_ifdata(use_csv=True)
    query_path = os.path.join(data_path, 'queries.txt')
    queries = query_viz.generate_queries_from_file(query_path)
    false_neg_path = os.path.join(data_path, 'false_negs.csv')
    false_negs = data_utils.get_false_negs(false_neg_path)
    export_path = os.path.join(out_path, 'all_images')
    with open(os.path.join(sg_path, 'sg_test_annotations.json')) as f:
        json_data = json.load(f)
    export_images(queries, if_data, false_negs, export_path, json_data)
