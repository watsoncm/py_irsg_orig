import os
import json

import data_pull as dp
import query_viz

with open('config.json') as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']


def save_query_texts(queries, if_data, path):
    with open(path, 'w') as f:
        for query_index, query in enumerate(queries):
            iqd = query_viz.ImageQueryData(query, query_index, 0, if_data)
            f.write('[{}]: {}\n'.format(query_index, iqd.get_query_text()))


if __name__ == '__main__':
    _, _, _, _, queries, if_data = dp.get_all_data(use_csv=True)
    query_path = os.path.join(out_path, 'query_texts.txt')
    save_query_texts(queries['simple_graphs'], if_data, query_path)
