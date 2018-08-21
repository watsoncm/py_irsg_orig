"""A basic helper script to poke around the data."""

import os
import json

import query_viz
import data_utils
import irsg_core.data_pull as dp
import irsg_core.image_fetch_utils as ifu
from config import get_config_path

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    data_path = cfg_data['file_paths']['mat_path']

data = dp.get_all_data(use_csv=True)
vgd, potentials, platt_mod, bin_mod, irsg_queries, ifdata = data

query_path = os.path.join(data_path, 'queries.txt')
queries = query_viz.generate_queries_from_file(query_path)


# TEST CODE BELOW

data = data_utils.get_partial_query_matches(
    vgd['vg_data_test'], queries)
old_data = ifu.get_partial_scene_matches(
    vgd['vg_data_test'], queries)

print(data)
print(old_data)

import pdb; pdb.set_trace()
