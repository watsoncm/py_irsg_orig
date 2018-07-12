import data_pull as dp

ifdata = dp.get_hdf_data()

ifdata.configure(17, None)
print(ifdata.object_detections)

dp.close_hdf()
