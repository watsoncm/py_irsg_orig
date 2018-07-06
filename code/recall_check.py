import os
import json
import numpy as np

import examples
import data_pull as dp
import image_fetch_wrappers as ifw
import image_fetch_utils as ifu
import image_fetch_plot as ifp

with open('config.json') as f:
    cfg_data = json.load(f)

out_path = cfg_data['file_paths']['output_path']


def recall_check(do_holdout=False, x_limit=-1, query_idx=None):
    vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
    images = vgd['vg_data_test']
    tp_queries = queries['simple_graphs']
    if query_idx is None:
        tp_queries = tp_queries[query_idx]

    tp_simple = ifu.get_partial_scene_matches(images, tp_queries)
    data_simple = [(os.path.join(out_path, 'original_simple/'), 'orignal')]
    ifp.r_at_k_plot_simple(data_simple, tp_simple, do_holdout=do_holdout,
                           x_limit=x_limit)


if __name__ == '__main__':
    recall_check(query_idx=list(range(60)))
