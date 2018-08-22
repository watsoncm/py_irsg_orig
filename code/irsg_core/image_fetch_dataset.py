import os
import json
from collections import namedtuple

import numpy as np

import image_fetch_utils as ifu
import gmm_utils
from config import get_config_path

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    csv_path = cfg_data['file_paths']['csv_path']


class ImageFetchDataset(object):
    def __init__(self, vg_data, potentials_data, platt_models,
                 relationship_models, base_image_path):
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

    def configure(self, image_index, sg_query):
        if image_index != self.current_image_num:
            self.current_image_num = image_index
            self.object_detections = ifu.get_object_detections(
                self.current_image_num, self.potentials_data,
                self.platt_models)
            self.attribute_detections = ifu.get_attribute_detections(
                self.current_image_num, self.potentials_data,
                self.platt_models)
            self.image_filename = self.base_image_path + os.path.basename(
                self.vg_data[self.current_image_num].image_path)

        if sg_query is not None or sg_query != self.current_sg_query:
            self.current_sg_query = sg_query
            self.per_object_attributes = ifu.get_object_attributes(
                self.current_sg_query)


class CSVImageFetchDataset(ImageFetchDataset):
    def __init__(self, dataset, split, vg_data, base_image_path, csv_path):
        super(CSVImageFetchDataset, self).__init__([], None, None, None,
                                                   base_image_path)
        self.dataset = dataset
        self.split = split
        self.csv_path = csv_path
        self.potentials_data = {}

        # get paths
        self.dataset_path = os.path.join(csv_path, 'datasets', dataset)
        self.split_path = os.path.join(self.dataset_path, split)
        self.obj_path = os.path.join(self.split_path, 'obj_files')
        self.attr_path = os.path.join(self.split_path, 'attr_files')
        self.rel_path = os.path.join(self.split_path, 'rel_files')

        # get class data
        classes, class_to_idx = self.read_csv_class_data()
        self.potentials_data['classes'] = classes
        self.potentials_data['class_to_idx'] = class_to_idx

        # get vg_data
        with open(os.path.join(self.split_path, 'index.txt')) as f:
            self.indices = [int(line) for line in f.read().splitlines()]
        self.vg_data = self.load_vg_data(vg_data)

        # get box and score data
        self.n_images = self.vg_data.shape[0]
        self.n_objs = len(os.listdir(self.obj_path))
        self.n_attrs = len(os.listdir(self.attr_path))
        empty_pots = [None for _ in range(self.n_images)]
        self.potentials_data['boxes'] = np.array(empty_pots)
        self.potentials_data['scores'] = np.array(empty_pots)
        self.loaded_cache = []

        # get attribute and object lists
        self.all_objs = [obj[4:] for obj in class_to_idx.keys()
                         if 'obj' in obj]
        self.all_attrs = [attr[4:] for attr in class_to_idx.keys()
                          if 'atr' in attr]

        # get relationship and platt models
        self.relationship_models = self.read_csv_rel_models()
        if self.split == 'train':
            self.platt_models = None
        else:
            self.platt_models = self.read_csv_platt_models()

    def load_vg_data(self, vg_data):
        if self.split == 'test':
            vg_key = 'vg_data_test'
        else:
            vg_key = 'vg_data_train'
        return vg_data[vg_key][self.indices]

    def read_csv_class_data(self):
        classes = np.loadtxt(os.path.join(csv_path, 'classes.csv'),
                             dtype='O', delimiter=',')
        class_to_idx = np.loadtxt(os.path.join(csv_path, 'class_to_idx.csv'),
                                  dtype=[('class', 'O'), ('idx', int)],
                                  delimiter=',')
        class_to_idx = dict(zip(class_to_idx['class'], class_to_idx['idx']))
        return classes.reshape(-1), class_to_idx

    def read_csv_rel_models(self):
        args = ['gmm_weights', 'gmm_mu', 'gmm_sigma']
        if self.split != 'train':
            args += ['platt_a', 'platt_b']
        GMM = namedtuple('GMM', args)
        rel_models = {}
        for model_name in os.listdir(self.rel_path):
            if self.split == 'train':
                gmm_data = gmm_utils.load_gmm_data(model_name, self.rel_path,
                                                   load_platt=False)
                gmm_weights, gmm_mu, gmm_sigma = gmm_data
                rel_models[model_name] = GMM(gmm_weights=gmm_weights,
                                             gmm_mu=gmm_mu,
                                             gmm_sigma=gmm_sigma)
            else:
                gmm_data = gmm_utils.load_gmm_data(model_name, self.rel_path,
                                                   load_platt=True)
                gmm_weights, gmm_mu, gmm_sigma, platt_a, platt_b = gmm_data
                rel_models[model_name] = GMM(gmm_weights=gmm_weights,
                                             gmm_mu=gmm_mu,
                                             gmm_sigma=gmm_sigma,
                                             platt_a=platt_a, platt_b=platt_b)
        return rel_models

    def read_csv_platt_models(self):
        platt_data = {}
        PlattModels = namedtuple('PlattModels', ['s_models'])
        ModelDict = namedtuple('ModelList', ['serialization'])
        Serialization = namedtuple('Serialization', ['keys', 'values'])
        for path, prefix in ((self.attr_path, 'atr'), (self.obj_path, 'obj')):
            for name in os.listdir(path):
                with open(os.path.join(path, name, 'platt.csv')) as f:
                    platt_a, platt_b = [float(v) for v in f.read().split(',')]
                platt_array = np.array([platt_a, platt_b, np.nan, np.nan])
                platt_data['{}:{}'.format(prefix, name)] = platt_array

        keys, values = zip(*platt_data.items())
        serialization = Serialization(keys=keys, values=values)
        s_models = ModelDict(serialization=serialization)
        return {'platt_models': PlattModels(s_models=s_models)}

    def load_data(self, name, is_obj, image_index):
        abs_index = self.indices[image_index]
        csv_name = 'irsg_{}.csv'.format(abs_index)
        obj_attr_path = self.obj_path if is_obj else self.attr_path
        csv_file_path = os.path.join(obj_attr_path, name, csv_name)

        # load relevant data and assign boxes
        data_array = np.loadtxt(csv_file_path, delimiter=',')
        self.potentials_data['boxes'][image_index] = data_array[:, :4]

        # create score array if necessary
        if self.potentials_data['scores'][image_index] is None:
            scores_shape = (data_array.shape[0], self.n_objs + self.n_attrs)
            self.potentials_data['scores'][image_index] = np.zeros(
                scores_shape)
        full_name = ('obj:' if is_obj else 'atr:') + name
        score_idx = self.potentials_data['class_to_idx'][full_name] - 1
        image_scores = self.potentials_data['scores'][image_index]
        image_scores[:, score_idx] = data_array[:, 4]

    def load_relevant_data(self, image_index, sg_query, load_all=False):
        if sg_query is not None or load_all:
            obj_names = (self.all_objs if load_all else
                         [np.array(obj.names).reshape(-1)[0]
                          for obj in sg_query.objects])
            attr_names = (self.all_attrs if load_all else
                          [attr_name for _, attr_name
                           in ifu.get_object_attributes(sg_query)])
            for obj in obj_names:
                if (obj, image_index) not in self.loaded_cache:
                    if ('obj:{}'.format(obj) in
                            self.potentials_data['classes'] or load_all):
                        self.load_data(obj, True, image_index)
                        self.loaded_cache.append((obj, image_index))
            for attr in attr_names:
                if (attr, image_index) not in self.loaded_cache:
                    if ('atr:{}'.format(attr) in
                            self.potentials_data['classes'] or load_all):
                        self.load_data(attr, False, image_index)
                        self.loaded_cache.append((attr, image_index))

    def configure(self, image_index, sg_query, load_all=False):
        if image_index != self.current_image_num:
            self.current_image_num = image_index
            self.load_relevant_data(image_index, sg_query,
                                    load_all=load_all)
            if sg_query is not None or load_all:
                self.object_detections = ifu.get_object_detections(
                    self.current_image_num, self.potentials_data,
                    self.platt_models, use_csv=True)
                self.attribute_detections = ifu.get_attribute_detections(
                    self.current_image_num, self.potentials_data,
                    self.platt_models, use_csv=True)
            self.image_filename = self.base_image_path + os.path.basename(
                self.vg_data[self.current_image_num].image_path)

        if sg_query is not None or sg_query != self.current_sg_query:
            self.current_sg_query = sg_query
            self.per_object_attributes = ifu.get_object_attributes(
                self.current_sg_query)
