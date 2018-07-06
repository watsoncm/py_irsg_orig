import os
import json
import numpy as np

import examples
import data_pull as dp
import image_fetch_wrappers as ifw
import image_fetch_utils as ifu
import image_fetch_plot as ifp


def generate_data():
    examples.ex5(range(150))


def recall_check(do_holdout=False, x_limit=-1):
    vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data()
    tp_simple = ifu.get_partial_scene_matches(vgd['vg_data_test'],
                                              queries['simple_graphs'])
    data_simple = [('/home/watsonc/py_irsg_orig/output/original_simple',
                    'orignal')]
    ifp.r_at_k_plot_simple(data_simple, tp_simple, do_holdout=do_holdout,
                           x_limit=x_limit)


if __name__ == '__main__':
    recall_check()
