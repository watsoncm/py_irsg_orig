import os
import json

import query_viz
import data_utils
import irsg_core.data_pull as dp
from image_query_data import ImageQueryData
from config import get_config_path

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    data_path = cfg_data['file_paths']['mat_path']
    out_path = cfg_data['file_paths']['output_path']


def save_query_counts(queries, if_data, false_negs, path):
    """Save the counts of the given queries to a file."""
    tp_simple = data_utils.get_partial_query_matches(
        if_data.vg_data, queries)
    for query_index, image_index in false_negs:
        tp_simple[query_index].append(image_index)

    with open(path, 'w') as f:
        for index, (query, image_index) in enumerate(zip(queries, tp_simple)):
            iqd = ImageQueryData(query, index, 0, if_data)
            f.write('{} ({} pos, {} neg)\n'.format(
                iqd.get_query_text(), len(image_index),
                len(if_data.vg_data) - len(image_index)))
            full_negs = [index for index in range(len(if_data.vg_data))
                         if all([index not in image_index
                                 for image_index in tp_simple])]
            print(full_negs)
        f.write('negatives for ALL queries: {}'.format(len(full_negs)))


if __name__ == '__main__':
    if_data = dp.get_ifdata(use_csv=True)
    query_path = os.path.join(data_path, 'queries.txt')
    queries = query_viz.generate_queries_from_file(query_path)
    false_neg_path = os.path.join(data_path, 'false_negs.csv')
    false_negs = data_utils.get_false_negs(false_neg_path)
    count_path = os.path.join(out_path, 'counts.txt')
    save_query_counts(queries, if_data, false_negs, count_path)
