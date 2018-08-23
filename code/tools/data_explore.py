"""A basic helper script to poke around the data."""

import pdb

import irsg_core.data_pull as dp

data = dp.get_all_data()
vgd, potentials, platt_mod, bin_mod, irsg_queries, ifdata = data

pdb.set_trace()
