import numpy as np
import json
import os.path

import image_fetch_utils as ifu
from config import get_config_path

with open(get_config_path()) as cfg_file:
    cfg_data = json.read(cfg_file)
    csv_datasets = cfg_data['csv_datasets']


class ImageFetchDataset(object):
    def __init__(self, split, vg_data, potentials_data, platt_models,
                 relationship_models, base_image_path):
        # TODO: implement logic to choose split (set vg_data to vg_data[...])
        self.split = split
        self.vg_data = vg_data
        self.potentials_data = potentials_data
        self.platt_models = platt_models
        self.relationship_models = relationship_models
        self.base_image_path = base_image_path

        self.current_image_num = -1
        self.object_detections = None
        self.attribute_detections = None
        self.per_object_attributes = None
        self.image_filename = ""
        self.current_sg_query = None

    def configure(self, test_image_num, sg_query):
        if test_image_num != self.current_image_num:
            self.current_image_num = test_image_num
            self.object_detections = ifu.get_object_detections(
                self.current_image_num, self.potentials_data,
                self.platt_models)
            self.attribute_detections = ifu.get_attribute_detections(
                self.current_image_num, self.potentials_data,
                self.platt_models)
            self.image_filename = self.base_image_path + os.path.basename(
                self.vg_data[self.current_image_num].image_path)

        if sg_query != self.current_sg_query:
            self.current_sg_query = sg_query
            self.per_object_attributes = ifu.get_object_attributes(
                self.current_sg_query)


# TODO: check if platt_models is okay to be None
class CSVImageFetchDataset(ImageFetchDataset):
    VALID_DATASETS = ('default', 'psu', 'psu-small')

    def __init__(self, dataset, split, vg_data, platt_models,
                 relationship_models, base_image_path, csv_path):
        super(CSVImageFetchDataset, self).__init__(vg_data, None, platt_models,
                                                   relationship_models,
                                                   base_image_path)
        self.dataset = dataset
        if self.dataset not in self.VALID_DATASETS:
            error_format = 'invalid dataset {}: valid values are {}'
            valid_sets = ', '.join(self.VALID_DATASETS)
            raise ValueError(error_format.format(self.dataset, valid_sets))
        self.split = split
        self.csv_path = csv_path
        self.potentials_data = {}

        # get class data
        classes = np.loadtxt(os.path.join(csv_path, 'classes.csv'), dtype='O',
                             delimiter=',')
        class_to_idx = np.loadtxt(os.path.join(csv_path, 'class_to_idx.csv'),
                                  dtype=[('class', 'O'), ('idx', int)],
                                  delimiter=',')
        class_to_idx = dict(zip(class_to_idx['class'], class_to_idx['idx']))
        self.potentials_data['classes'] = classes.reshape(-1)
        self.potentials_data['class_to_idx'] = class_to_idx

        # get box and score data
        self.n_images = self.vg_data.shape[0]
        self.n_objs = len(os.listdir(os.path.join(csv_path, 'obj_files')))
        self.n_attrs = len(os.listdir(os.path.join(csv_path, 'attr_files')))
        empty_array = [None for _ in range(self.n_images)]
        self.potentials_data['boxes'] = np.array(empty_array)
        self.potentials_data['scores'] = np.array(empty_array)
        self.loaded_cache = []

        # get attribute and object lists
        self.all_objs = [obj[4:] for obj in class_to_idx.keys()
                         if 'obj' in obj]
        self.all_attrs = [attr[4:] for attr in class_to_idx.keys()
                          if 'atr' in attr]

        if self.dataset != 'default':
            self.relationship_models = self.read_csv_rel_models()
            self.platt_models = self.read_csv_platt_models()

    def load_data(self, name, is_obj, test_image_num):
        csv_name = 'irsg_{}.csv'.format(test_image_num)
        dir_name = 'obj_files' if is_obj else 'attr_files'
        csv_file_path = os.path.join(self.csv_path, dir_name, name, csv_name)

        # load relevant data and assign boxes
        data_array = np.loadtxt(csv_file_path, delimiter=',')
        self.potentials_data['boxes'][test_image_num] = data_array[:, :4]

        # create score array if necessary
        if self.potentials_data['scores'][test_image_num] is None:
            scores_shape = (data_array.shape[0], self.n_objs + self.n_attrs)
            self.potentials_data['scores'][test_image_num] = np.zeros(
                scores_shape)
        full_name = ('obj:' if is_obj else 'atr:') + name
        score_idx = self.potentials_data['class_to_idx'][full_name] - 1
        image_scores = self.potentials_data['scores'][test_image_num]
        image_scores[:, score_idx] = data_array[:, 4]

    def load_relevant_data(self, test_image_num, sg_query, load_all=False):
        if sg_query is not None or load_all:
            obj_names = (self.all_objs if load_all else
                         [np.array(obj.names).reshape(-1)[0]
                          for obj in sg_query.objects])
            attr_names = (self.all_attrs if load_all else
                          [attr_name for _, attr_name
                           in ifu.get_object_attributes(sg_query)])
            for obj in obj_names:
                if (obj, test_image_num) not in self.loaded_cache:
                    if ('obj:{}'.format(obj) in
                            self.potentials_data['classes'] or load_all):
                        self.load_data(obj, True, test_image_num)
                        self.loaded_cache.append((obj, test_image_num))
            for attr in attr_names:
                if (attr, test_image_num) not in self.loaded_cache:
                    if ('atr:{}'.format(attr) in
                            self.potentials_data['classes'] or load_all):
                        self.load_data(attr, False, test_image_num)
                        self.loaded_cache.append((attr, test_image_num))

    def configure(self, test_image_num, sg_query, load_all=False):
        if test_image_num != self.current_image_num:
            self.current_image_num = test_image_num
            self.load_relevant_data(test_image_num, sg_query,
                                    load_all=load_all)
            self.object_detections = ifu.get_object_detections(
                self.current_image_num, self.potentials_data,
                self.platt_models, use_csv=True)
            self.attribute_detections = ifu.get_attribute_detections(
                self.current_image_num, self.potentials_data,
                self.platt_models, use_csv=True)
            self.image_filename = self.base_image_path + os.path.basename(
                self.vg_data[self.current_image_num].image_path)

        if sg_query != self.current_sg_query:
            self.current_sg_query = sg_query
            self.per_object_attributes = ifu.get_object_attributes(
                self.current_sg_query)
