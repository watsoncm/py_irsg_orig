import data_pull as dp

vgd, potentials, platt_mod, bin_mod, queries, ifdata = dp.get_all_data(use_csv=True)
query = vgd['vg_data_test'][0].annotations
import pdb; pdb.set_trace()
ifdata.configure(17, vgd['vg_data_test'][0].annotations)
