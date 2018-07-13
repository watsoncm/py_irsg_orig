import numpy as np
import collections
import image_fetch_utils as ifu
import os.path


class ImageFetchDataset(object):
    def __init__(self, vg_data, potentials_data, platt_models, relationship_models, base_image_path):
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
            self.object_detections = ifu.get_object_detections(self.current_image_num, self.potentials_data, self.platt_models)
            self.attribute_detections = ifu.get_attribute_detections(self.current_image_num, self.potentials_data, self.platt_models)
            self.image_filename = self.base_image_path + os.path.basename(self.vg_data[self.current_image_num].image_path)

        if sg_query != self.current_sg_query:
            self.current_sg_query = sg_query
            self.per_object_attributes = ifu.get_object_attributes(self.current_sg_query)


class CSVImageFetchDataset(ImageFetchDataset):
    def __init__(self, vg_data, platt_models, relationship_models, base_image_path, csv_path):
        super(CSVImageFetchDataset, self).__init__(vg_data, None, platt_models,
                                                   relationship_models,
                                                   base_image_path)
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
        self.potentials_data['boxes'] = np.array([None for _ in range(self.n_images)])
        self.potentials_data['scores'] = np.array([None for _ in range(self.n_images)])

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
            self.potentials_data['scores'][test_image_num] = np.zeros(scores_shape)
        full_name = ('obj:' if is_obj else 'atr:') + name
        score_idx = self.potentials_data['class_to_idx'][full_name] - 1  # zero-index
        self.potentials_data['scores'][test_image_num][:, score_idx] = data_array[:, 4]


    def load_relevant_data(self, test_image_num, sg_query):
        if sg_query is not None:
            for obj in sg_query.objects:
                name = np.array(obj.names).reshape(-1)[0]
                if 'obj:{}'.format(name) in self.potentials_data['classes']:
                    print('obj: {}'.format(name))
                    self.load_data(name, True, test_image_num)
            for attr in sg_query.unary_triples:
                name = np.array(obj.names).reshape(-1)[0]
                if 'atr:{}'.format(name) in self.potentials_data['classes']:
                    print('attr: {}'.format(name))
                    self.load_data(name, False, test_image_num)

    def configure(self, test_image_num, sg_query):
        if test_image_num != self.current_image_num:
            self.current_image_num = test_image_num
            self.load_relevant_data(test_image_num, sg_query)
            self.object_detections = ifu.get_object_detections(self.current_image_num, self.potentials_data, self.platt_models, use_csv=True)
            self.attribute_detections = ifu.get_attribute_detections(self.current_image_num, self.potentials_data, self.platt_models, use_csv=True)
            self.image_filename = self.base_image_path + os.path.basename(self.vg_data[self.current_image_num].image_path)

        if sg_query != self.current_sg_query:
            self.current_sg_query = sg_query
            self.per_object_attributes = ifu.get_object_attributes(self.current_sg_query)
