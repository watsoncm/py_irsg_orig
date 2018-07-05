import json
import numpy as np

import data_pull as dp
import image_fetch_wrappers as ifw
import image_fetch_plot as ifp

with open('config.json') as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']


def recall_check(num_samples=150, gm_method='original'): 
    vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
    query_idxs = np.random.choice(queries['simple_graphs'].size, 
                                  size=num_samples, replace=False)
    batch_path = out_path + gm_method + '_recall/'
    for query_idx in query_idxs:
        query = queries['simple_graphs'][query_idx].annotations
        ifw.image_batch(query, query_idx, ifdata, batch_path, 
                        gm_method=gm_method, gen_plots=False)


if __name__ == '__main__':
    recall_check()
