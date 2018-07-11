import time

import data_pull as dp

t1 = time.time()
data = dp.get_all_data(use_hdf=True, reload=True)
t2 = time.time()
data = dp.get_all_data(use_hdf=False, reload=True)
t3 = time.time()

print('From HDF: {:.3f}s'.format(t2-t1))
print('From MAT: {:.3f}s'.format(t3-t2))
