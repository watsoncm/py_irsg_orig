import os
import json

import query_viz
import data_utils
import irsg_core.data_pull as dp
from config import get_config_path

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']


def save_query_texts(queries, if_data, path):
    """Save the texts of the given queries to a file."""
    tp_simple = data_utils.get_partial_query_matches(
        if_data.vg_data, queries)

    with open(path, 'w') as f:
        for index, (query, image_index) in enumerate(zip(queries, tp_simple)):
            iqd = query_viz.ImageQueryData(query, index, 0, if_data)
            f.write('[{}] (count: {}): {}\n'.format(
                index, len(image_index), iqd.get_query_text()))


if __name__ == '__main__':
    _, _, _, _, queries, if_data = dp.get_all_data(use_csv=True)
    query_path = os.path.join(out_path, 'query_texts.txt')
    save_query_texts(queries['simple_graphs'], if_data, query_path)
