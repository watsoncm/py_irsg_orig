import json

import numpy as np
import matplotlib.pyplot as plt

import data_pull as dp
import image_fetch_core as ifc
import image_fetch_utils as ifu

with open('config.json') as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']


class NotInitializedException(Exception):
    pass


class ImageQueryData(object):
    def __init__(self, query, query_id, image_id, if_data,
                 image_sub_id=None, image_obj_id=None):
        self.query = query
        self.query_id = query_id
        query_triple = self.make_array(query.annotations.binary_triples)[0]
        self.query_sub_id = query_triple.subject
        self.query_obj_id = query_triple.object
        query_sub_data = query.annotations.objects[self.query_sub_id]
        query_obj_data = query.annotations.objects[self.query_obj_id]
        self.query_sub = self.make_array(query_sub_data.names)[0]
        self.query_pred = query_triple.predicate
        self.query_obj = self.make_array(query_obj_data.names)[0]
        attr_triples = self.make_array(self.query.annotations.unary_triples)
        attributes = self.get_attributes(attr_triples, self.query_sub_id,
                                         self.query_obj_id)
        self.query_sub_attrs, self.query_obj_attrs = attributes

        self.image = if_data.vg_data[image_id]
        self.image_id = image_id
        if image_sub_id is None or image_obj_id is None:
            self.image_sub_id, self.image_obj_id = self.get_image_ids()
            if self.image_sub_id is None or self.image_obj_id is None:
                raise ValueError('image incompatible with query')
        else:
            self.image_sub_id = image_sub_id
            self.image_obj_id = image_obj_id

        self.if_data = if_data
        self.if_data.configure(self.image_id, self.query.annotations)
        self.boxes = if_data.potentials_data['boxes'][self.image_id]
        image_objects = self.image.annotations.objects
        image_sub_mat_bbox = image_objects[self.image_sub_id].bbox
        image_obj_mat_bbox = image_objects[self.image_obj_id].bbox
        self.image_sub_bbox = self.make_bbox(image_sub_mat_bbox)
        self.image_obj_bbox = self.make_bbox(image_obj_mat_bbox)
        self.image_scores = self.if_data.potentials_data['scores'][self.image_id]
        self.class_to_idx = self.if_data.potentials_data['class_to_idx']
        self.initialized_ = False

    @staticmethod
    def make_array(items_or_single):
        return np.array(items_or_single).reshape(-1)

    @staticmethod
    def make_bbox(mat_bbox):
        return np.array([mat_bbox.x, mat_bbox.y, mat_bbox.w, mat_bbox.h])

    @staticmethod
    def get_attributes(attr_triples, subject_id, object_id):
        sub_attrs, obj_attrs = [], []
        for attr_triple in attr_triples:
            if attr_triple.subject == subject_id:
                sub_attrs.append(attr_triple.object)
            elif attr_triple.subject == object_id:
                obj_attrs.append(attr_triple.object)
        return sub_attrs, obj_attrs

    def triple_matches(self, trip_sub, trip_pred, trip_obj,
                       sub_attrs, obj_attrs):
        has_sub_attrs = np.all([sub_attr in sub_attrs
                                for sub_attr in self.query_sub_attrs])
        has_obj_attrs = np.all([obj_attr in obj_attrs
                                for obj_attr in self.query_obj_attrs])
        return (trip_sub == self.query_sub and
                trip_pred == self.query_pred and
                trip_obj == self.query_obj and
                has_sub_attrs and
                has_obj_attrs)

    def get_image_ids(self):
        unary_triples = self.make_array(self.image.annotations.unary_triples)
        binary_triples = self.make_array(self.image.annotations.binary_triples)
        for triple in binary_triples:
            trip_sub, trip_pred, trip_obj = triple.text
            sub_attrs, obj_attrs = self.get_attributes(unary_triples,
                                                       triple.subject,
                                                       triple.object)
            if (self.triple_matches(trip_sub, trip_pred, trip_obj,
                                    sub_attrs, obj_attrs)):
                return triple.subject, triple.object
        return None, None

    def get_query_text(self):
        query_sub = self.make_array(self.subject.names)[0]
        query_obj = self.make_array(query_objs[triple.object].names)[0]
        sub_prefix = (', '.join(self.query_names['subject_attrs']) + ' '
                      if len(self.query_names['subject_attrs']) > 0 else '')
        obj_prefix = (', '.join(self.query_names['object_attrs']) + ' '
                      if len(self.query_names['object_attrs']) > 0 else '')
        return '{}{} {} {}{}'.format(sub_prefix, self.query_names['subject'],
                                     self.query_names['relationship'],
                                     obj_prefix, self.query_names['object'])

    @staticmethod
    def get_iou(bbox_a, bbox_b):
        bbox_a_x, bbox_a_y, bbox_a_w, bbox_a_h = bbox_a
        bbox_b_x, bbox_b_y, bbox_b_w, bbox_b_h = bbox_b

        # convert to (x1, x2, y1, y2)
        bbox_a_x1 = bbox_a_x - bbox_a_w / 2.0
        bbox_a_x2 = bbox_a_x + bbox_a_w / 2.0
        bbox_a_y1 = bbox_a_y - bbox_a_h / 2.0
        bbox_a_y2 = bbox_a_y + bbox_a_h / 2.0

        bbox_b_x1 = bbox_b_x - bbox_b_w / 2.0
        bbox_b_x2 = bbox_b_x + bbox_b_w / 2.0
        bbox_b_y1 = bbox_b_y - bbox_b_h / 2.0
        bbox_b_y2 = bbox_b_y + bbox_b_h / 2.0

        # find corners of intersection
        bbox_in_x1 = max((bbox_a_x1, bbox_b_x1))
        bbox_in_x2 = min((bbox_a_x2, bbox_b_x2))
        bbox_in_y1 = max((bbox_a_y1, bbox_b_y1))
        bbox_in_y2 = min((bbox_a_y2, bbox_b_y2))

        # if the intersection is zero, we're done
        in_w = bbox_in_x2 - bbox_in_x1
        in_h = bbox_in_y2 - bbox_in_y1
        if in_w <= 0.0 or in_h <= 0.0:
            return 0.0

        # otherwise, return answer
        in_area = in_w * in_h
        a_area = bbox_a_w * bbox_a_h
        b_area = bbox_b_w * bbox_b_h
        return in_area / (a_area + b_area - in_area)

    def get_gt_closest(self):
        sub_ious = [self.get_iou(box, self.image_sub_bbox)
                    for box in self.boxes]
        obj_ious = [self.get_iou(box, self.image_obj_bbox)
                    for box in self.boxes]
        sub_idx = np.argmax(sub_ious)
        obj_idx = np.argmax(obj_ious)
        return (sub_idx, obj_idx, sub_ious[sub_idx], obj_ious[obj_idx])

    def get_model_boxes(self):
        gm, tracker = ifc.generate_pgm(self.if_data, verbose=False)
        _, best_matches, _ = ifc.do_inference(gm)
        return best_matches[0], best_matches[1]

    def get_obj_scores(self):
        sub_pot_id = self.class_to_idx['obj:' + self.query_sub] - 1
        obj_pot_id = self.class_to_idx['obj:' + self.query_obj] - 1
        model_sub_pot = self.image_scores[self.model_sub_bbox_id, sub_pot_id]
        model_obj_pot = self.image_scores[self.model_obj_bbox_id, obj_pot_id]
        gt_sub_pot = self.image_scores[self.close_sub_bbox_id, sub_pot_id]
        gt_obj_pot = self.image_scores[self.close_obj_bbox_id, obj_pot_id]
        return model_sub_pot, model_obj_pot, gt_sub_pot, gt_obj_pot

    def get_attr_scores(self):
        sub_attr_pot_ids = [self.class_to_idx['atr:' + attr] - 1
                            for attr in self.query_sub_attrs]
        obj_attr_pot_ids = [self.class_to_idx['atr:' + attr] - 1
                            for attr in self.query_obj_attrs]
        model_sub_attr_pots = []
        model_obj_attr_pots = []
        gt_sub_attr_pots = []
        gt_obj_attr_pots = []

        for pot_id in sub_attr_pot_ids:
            model_pot = self.image_scores[self.model_sub_bbox_id, pot_id]
            model_sub_attr_pots.append(model_pot)
            gt_pot = self.image_scores[self.close_sub_bbox_id, pot_id]
            gt_sub_attr_pots.append(gt_pot)
        for pot_id in obj_attr_pot_ids:
            model_pot = self.image_scores[self.model_obj_bbox_id, pot_id]
            model_obj_attr_pots.append(model_pot)
            gt_pot = self.image_scores[self.close_obj_bbox_id, pot_id]
            gt_obj_attr_pots.append(gt_pot)
        return (model_sub_attr_pots, model_obj_attr_pots,
                gt_sub_attr_pots, gt_obj_attr_pots)

    def get_pred_model(self):
        pred_models = self.if_data.relationship_models
        clean_pred = self.query_pred.replace(' ', '_')
        bin_trip_key = '{}_{}_{}'.format(self.query_sub, clean_pred,
                                         self.query_obj)
        if bin_trip_key not in pred_models:
            bin_trip_key = '*_{}_*'.format(clean_pred)
        try:
            return pred_models[bin_trip_key], bin_trip_key
        except KeyError:
            return None, None

    def score_box_pair(self, subject_box, object_box):
        sub_center = (subject_box[0] + 0.5 * subject_box[2],
                      subject_box[1] + 0.5 * subject_box[3])
        obj_center = (object_box[0] + 0.5 * object_box[2],
                      object_box[1] + 0.5 * object_box[3])
        rel_center = ((sub_center[0] - obj_center[0]) / subject_box[2],
                      (sub_center[1] - obj_center[1]) / subject_box[3])
        rel_dims = (object_box[2] / subject_box[2],
                    object_box[3] / subject_box[3])
        features = np.vstack((rel_center[0], rel_center[1],
                              rel_dims[1], rel_dims[0])).T

        scores = ifc.gmm_pdf(features, self.pred_model.gmm_weights,
                             self.pred_model.gmm_mu, self.pred_model.gmm_sigma)
        eps = np.finfo(np.float).eps
        scores = np.log(eps + scores)
        sig_scores = 1.0 / (1.0 + np.exp(self.pred_model.platt_a * scores +
                                         self.pred_model.platt_b))
        return scores[0], sig_scores[0]

    def get_pred_scores(self):
        model_sub_bbox = self.boxes[self.model_sub_bbox_id]
        model_obj_bbox = self.boxes[self.model_obj_bbox_id]
        close_sub_bbox = self.boxes[self.close_sub_bbox_id]
        close_obj_bbox = self.boxes[self.close_obj_bbox_id]
        model_pred_score = self.score_box_pair(model_sub_bbox, model_obj_bbox)
        close_pred_score = self.score_box_pair(close_sub_bbox, close_obj_bbox)
        return model_pred_score, close_pred_score

    def get_heatmaps(self):
        return None, None  # TODO: implement

    def compute_plot_data(self):
        (self.close_sub_bbox_id, self.close_obj_bbox_id,
         self.sub_iou, self.obj_iou) = self.get_gt_closest()
        self.model_sub_bbox_id, self.model_obj_bbox_id = self.get_model_boxes()
        (self.model_sub_pot, self.model_obj_pot,
         self.gt_sub_pot, self.gt_obj_pot) = self.get_obj_scores()
        (self.model_sub_attr_pots, self.model_obj_attr_pots,
         self.gt_sub_attr_pots, self.gt_sub_attr_pots) = self.get_attr_scores()
        self.pred_model, self.pred_model_name = self.get_pred_model()
        if self.pred_model is None:
            raise ValueError('no relevant relationship model exists')
        self.model_pred_score, self.gt_pred_score = self.get_pred_scores()
        self.model_heatmap, self.gt_heatmap = self.get_heatmaps()
        self.initialized_ = True

    def generate_plot(self, save_path=None):
        if not self.initialized_:
            raise NotInitializedException('ImageQueryData not initialized')

        fig, ax = plt.subplots(1, 2)
        plt.xticks([])
        plt.yticks([])

        # plt.title()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=175)


if __name__ == '__main__':
    _, _, _, _, queries, if_data = dp.get_all_data(use_csv=True)
    tp_data = ifu.get_partial_scene_matches(if_data.vg_data,
                                            queries['simple_graphs'])
    iqd = ImageQueryData(queries['simple_graphs'][0], 0, tp_data[0][0],
                         if_data)
    iqd.compute_plot_data()

    import pdb
    pdb.set_trace()

    iqd.generate_plot()
