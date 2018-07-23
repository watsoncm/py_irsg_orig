import os
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

import data_pull as dp
import image_fetch_core as ifc
import image_fetch_utils as ifu
import gmm_viz

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
        potentials = self.if_data.potentials_data
        self.boxes = potentials['boxes'][self.image_id]
        image_objects = self.image.annotations.objects
        image_sub_mat_bbox = image_objects[self.image_sub_id].bbox
        image_obj_mat_bbox = image_objects[self.image_obj_id].bbox
        self.image_sub_bbox = self.make_bbox(image_sub_mat_bbox)
        self.image_obj_bbox = self.make_bbox(image_obj_mat_bbox)
        self.image_scores = potentials['scores'][self.image_id]
        self.class_to_idx = potentials['class_to_idx']
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
        sub_prefix = (', '.join(self.query_sub_attrs) + ' '
                      if len(self.query_sub_attrs) > 0 else '')
        obj_prefix = (', '.join(self.query_obj_attrs) + ' '
                      if len(self.query_obj_attrs) > 0 else '')
        return '{}{} {} {}{}'.format(sub_prefix, self.query_sub,
                                     self.query_pred,
                                     obj_prefix, self.query_obj)

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
        gm, _ = ifc.generate_pgm(self.if_data, verbose=False)
        energy, best_matches, _ = ifc.do_inference(gm)
        print(energy)
        return best_matches[0], best_matches[1]

    def get_obj_scores(self):
        sub_pot_id = self.class_to_idx['obj:' + self.query_sub] - 1
        obj_pot_id = self.class_to_idx['obj:' + self.query_obj] - 1
        model_sub_pot = self.image_scores[self.model_sub_bbox_id, sub_pot_id]
        model_obj_pot = self.image_scores[self.model_obj_bbox_id, obj_pot_id]
        close_sub_pot = self.image_scores[self.close_sub_bbox_id, sub_pot_id]
        close_obj_pot = self.image_scores[self.close_obj_bbox_id, obj_pot_id]
        log_pots = [-np.log(pot) for pot in (model_sub_pot, model_obj_pot,
                                             close_sub_pot, close_obj_pot)]
        return log_pots

    def get_attr_scores(self):
        sub_attr_pot_ids = [self.class_to_idx['atr:' + attr] - 1
                            for attr in self.query_sub_attrs]
        obj_attr_pot_ids = [self.class_to_idx['atr:' + attr] - 1
                            for attr in self.query_obj_attrs]
        model_sub_attr_pots = []
        model_obj_attr_pots = []
        close_sub_attr_pots = []
        close_obj_attr_pots = []

        for pot_id in sub_attr_pot_ids:
            model_pot = self.image_scores[self.model_sub_bbox_id, pot_id]
            model_sub_attr_pots.append(-np.log(model_pot))
            gt_pot = self.image_scores[self.close_sub_bbox_id, pot_id]
            close_sub_attr_pots.append(-np.log(gt_pot))
        for pot_id in obj_attr_pot_ids:
            model_pot = self.image_scores[self.model_obj_bbox_id, pot_id]
            model_obj_attr_pots.append(-np.log(model_pot))
            gt_pot = self.image_scores[self.close_obj_bbox_id, pot_id]
            close_obj_attr_pots.append(-np.log(gt_pot))
        return (model_sub_attr_pots, model_obj_attr_pots,
                close_sub_attr_pots, close_obj_attr_pots)

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
                              rel_dims[0], rel_dims[1])).T

        scores = ifc.gmm_pdf(features, self.pred_model.gmm_weights,
                             self.pred_model.gmm_mu, self.pred_model.gmm_sigma)
        eps = np.finfo(np.float).eps
        scores = np.log(eps + scores)
        sig_scores = 1.0 / (1.0 + np.exp(self.pred_model.platt_a * scores +
                                         self.pred_model.platt_b))
        return scores[0], -np.log(sig_scores[0])

    def get_pred_scores(self):
        model_sub_bbox = self.boxes[self.model_sub_bbox_id]
        model_obj_bbox = self.boxes[self.model_obj_bbox_id]
        close_sub_bbox = self.boxes[self.close_sub_bbox_id]
        close_obj_bbox = self.boxes[self.close_obj_bbox_id]
        model_pred_scores = self.score_box_pair(model_sub_bbox, model_obj_bbox)
        close_pred_scores = self.score_box_pair(close_sub_bbox, close_obj_bbox)
        gt_pred_scores = self.score_box_pair(self.image_sub_bbox,
                                             self.image_obj_bbox)
        return model_pred_scores, close_pred_scores, gt_pred_scores

    def get_heatmaps(self, n_samples=1000):
        gmm = gmm_viz.get_gmm(self.pred_model)
        samples = gmm.sample(n_samples)[0]
        model_sub_bbox = self.boxes[self.model_sub_bbox_id]
        model_obj_bbox = self.boxes[self.model_obj_bbox_id]
        box_maps = []
        for sub_bbox, obj_bbox in ((model_sub_bbox, model_obj_bbox),
                                   (self.image_sub_bbox, self.image_obj_bbox)):
            # Figure out width and height
            box_map = np.zeros((self.image.image_width,
                                self.image.image_height))
            obj_w = samples[:, 2] * sub_bbox[2]
            obj_h = samples[:, 3] * sub_bbox[3]

            # Calculate center x and y
            sub_bbox_center_x = (sub_bbox[0] + 0.5 * sub_bbox[2])
            sub_bbox_center_y = (sub_bbox[1] + 0.5 * sub_bbox[3])
            obj_x = (sub_bbox_center_x - sub_bbox[2] * samples[:, 0] -
                     0.5 * obj_w)
            obj_y = (sub_bbox_center_y - sub_bbox[3] * samples[:, 1] -
                     0.5 * obj_h)

            # Blit all samples
            obj_samples = np.vstack((obj_x, obj_y, obj_w, obj_h)).T
            for obj_sample in obj_samples:
                gmm_viz.blit_sample(box_map, obj_sample)
            box_maps.append(box_map)
        return box_maps

    def get_image_array(self):
        image_name = os.path.basename(self.image.image_path)
        image_path = os.path.join(if_data.base_image_path, image_name)
        pil_image = Image.open(image_path).convert("L")
        return np.array(pil_image)

    def get_total_pots(self):
        model_total_pot = (self.model_sub_pot + self.model_obj_pot +
                           self.model_pred_score)
        model_addr_total = sum(self.model_sub_attr_pots +
                               self.model_obj_attr_pots)
        model_total_pot += model_addr_total

        gt_total_pot = (self.close_sub_pot + self.close_obj_pot +
                        self.gt_pred_score)
        close_total_pot = (self.close_sub_pot + self.close_obj_pot +
                           self.close_pred_score)
        close_addr_total = sum(self.close_sub_attr_pots +
                               self.close_obj_attr_pots)
        gt_total_pot += close_addr_total
        close_total_pot += close_addr_total
        return model_total_pot, close_total_pot, gt_total_pot

    def compute_plot_data(self):
        (self.close_sub_bbox_id, self.close_obj_bbox_id,
         self.sub_iou, self.obj_iou) = self.get_gt_closest()
        self.model_sub_bbox_id, self.model_obj_bbox_id = self.get_model_boxes()
        (self.model_sub_pot, self.model_obj_pot,
         self.close_sub_pot, self.close_obj_pot) = self.get_obj_scores()
        (self.model_sub_attr_pots, self.model_obj_attr_pots,
         self.close_sub_attr_pots,
         self.close_obj_attr_pots) = self.get_attr_scores()
        self.pred_model, self.pred_model_name = self.get_pred_model()
        if self.pred_model is None:
            raise ValueError('no relevant relationship model exists')
        ((self.model_raw_pred_score, self.model_pred_score),
         (self.close_raw_pred_score, self.close_pred_score),
         (self.gt_raw_pred_score, self.gt_pred_score)) = self.get_pred_scores()
        self.model_heatmap, self.gt_heatmap = self.get_heatmaps()
        self.image_array = self.get_image_array()
        self.query_text = self.get_query_text()
        (self.model_total_pot, self.gt_total_pot,
         self.close_total_pot) = self.get_total_pots()
        self.initialized_ = True

    @staticmethod
    def get_rect(bbox, params):
        return patches.Rectangle((bbox[0], bbox[1]),
                                 bbox[2], bbox[3], **params)

    def generate_plot(self, save_path=None):
        if not self.initialized_:
            raise NotInitializedException('ImageQueryData not initialized')

        # Set up plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.axis('off')
        ax2.axis('off')
        ax1.imshow(self.image_array, cmap='gray')
        ax2.imshow(self.image_array, cmap='gray')

        # Draw heatmaps
        model_heatmap_blur = gaussian_filter(self.model_heatmap, sigma=7)
        gt_heatmap_blur = gaussian_filter(self.gt_heatmap, sigma=7)
        ax1.imshow(model_heatmap_blur.T, alpha=0.4)
        ax2.imshow(gt_heatmap_blur.T, alpha=0.4)

        # Prepare parameters
        ax1_params = {'bbox': {'facecolor': 'black', 'alpha': 0.8},
                      'color': 'white',
                      'size': 9,
                      'transform': ax1.transAxes}
        ax2_params = {'bbox': {'facecolor': 'black', 'alpha': 0.8},
                      'color': 'white',
                      'size': 9,
                      'transform': ax2.transAxes}
        sub_params = {'linewidth': 3,
                      'edgecolor': 'red',
                      'facecolor': 'none'}
        obj_params = {'linewidth': 3,
                      'edgecolor': 'green',
                      'facecolor': 'none'}
        alt_sub_params = dict(sub_params.items() + [('linestyle', 'dashed')])
        alt_obj_params = dict(obj_params.items() + [('linestyle', 'dashed')])

        # Add descriptive text
        ax1.text(0.02, 0.02, 'IRSG grounding'.format(self.query_text),
                 horizontalalignment='left', verticalalignment='bottom',
                 **ax1_params)
        ax2.text(0.02, 0.02, 'Ground truth'.format(self.query_text),
                 horizontalalignment='left', verticalalignment='bottom',
                 **ax2_params)
        ax1.text(0.98, 0.02, 'Query: "{}"'.format(self.query_text),
                 horizontalalignment='right', verticalalignment='bottom',
                 **ax1_params)
        ax2.text(0.98, 0.02, 'Query: "{}"'.format(self.query_text),
                 horizontalalignment='right', verticalalignment='bottom',
                 **ax2_params)

        # Add bounding boxes
        model_sub_bbox = self.boxes[self.model_sub_bbox_id]
        model_obj_bbox = self.boxes[self.model_obj_bbox_id]
        model_sub_patch = self.get_rect(model_sub_bbox, sub_params)
        model_obj_patch = self.get_rect(model_obj_bbox, obj_params)
        ax1.add_patch(model_sub_patch)
        ax1.add_patch(model_obj_patch)

        gt_sub_patch = self.get_rect(self.image_sub_bbox, sub_params)
        gt_obj_patch = self.get_rect(self.image_obj_bbox, obj_params)
        ax2.add_patch(gt_sub_patch)
        ax2.add_patch(gt_obj_patch)

        close_sub_bbox = self.boxes[self.close_sub_bbox_id]
        close_obj_bbox = self.boxes[self.close_obj_bbox_id]
        close_sub_patch = self.get_rect(close_sub_bbox, alt_sub_params)
        close_obj_patch = self.get_rect(close_obj_bbox, alt_obj_params)
        ax2.add_patch(close_sub_patch)
        ax2.add_patch(close_obj_patch)

        # Add IOU information
        iou_format = ('IOUs acceptable? {}\nSubject IOU: {:.3f}\n'
                      'Object IOU: {:.3f}')
        iou_ok = self.sub_iou >= 0.5 and self.obj_iou >= 0.5
        iou_text = iou_format.format(iou_ok, self.sub_iou, self.obj_iou)
        ax2.text(0.98, 0.98, iou_text, horizontalalignment='right',
                 verticalalignment='top', **ax2_params)

        # Extract attribute energy text
        attr_format = '        {} -> {}: {:.6f}'
        model_attr_lines, gt_attr_lines = [], []
        for attr, model_pot, gt_pot in zip(self.query_sub_attrs,
                                           self.model_sub_attr_pots,
                                           self.close_sub_attr_pots):
            model_line = attr_format.format(attr, self.query_sub, model_pot)
            gt_line = attr_format.format(attr, self.query_sub, gt_pot)
            model_attr_lines.append(model_line)
            gt_attr_lines.append(gt_line)
        for attr, model_pot, gt_pot in zip(self.query_obj_attrs,
                                           self.model_obj_attr_pots,
                                           self.close_obj_attr_pots):
            model_line = attr_format.format(attr, self.query_obj, model_pot)
            gt_line = attr_format.format(attr, self.query_obj, gt_pot)
            model_attr_lines.append(model_line)
            gt_attr_lines.append(gt_line)

        # Add energy information
        model_score_format = ('Subject ({}) potential: {:.6f}\n'
                              'Object ({}) potential: {:.6f}\n'
                              'Relationship ({}) potential: {:.6f}\n'
                              '(Pre-Platt scaling: {:.6f})\n'
                              'Attribute potentials:\n{}\n'
                              'Total potential: {:.6f}')
        gt_score_format = ('Subject ({}) potential (closest box): {:.6f}\n'
                           'Object ({}) potential (closest box): {:.6f}\n'
                           'Relationship ({}) potential: {:.6f}\n'
                           '(Pre-Platt scaling: {:.6f})\n'
                           'Relationship potential (closest boxes): {:.6f}\n'
                           '(Pre-Platt scaling: {:.6f})\n'
                           'Attribute potentials (closest boxes):\n{}\n'
                           'Total potential (using closest boxes): {:.6f}\n'
                           '(Using closest boxes except for relation: {:.6f})')
        model_score_parts = (self.query_sub, self.model_sub_pot,
                             self.query_obj, self.model_obj_pot,
                             self.pred_model_name, self.model_pred_score,
                             self.model_raw_pred_score,
                             '\n'.join(model_attr_lines), self.model_total_pot)
        gt_score_parts = (self.query_sub, self.close_sub_pot,
                          self.query_obj, self.close_obj_pot,
                          self.pred_model_name, self.gt_pred_score,
                          self.gt_raw_pred_score, self.close_pred_score,
                          self.close_raw_pred_score, '\n'.join(gt_attr_lines),
                          self.close_total_pot, self.gt_total_pot)
        model_score_text = model_score_format.format(*model_score_parts)
        gt_score_text = gt_score_format.format(*gt_score_parts)
        ax1.text(0.02, 0.98, model_score_text, horizontalalignment='left',
                 verticalalignment='top', **ax1_params)
        ax2.text(0.02, 0.98, gt_score_text, horizontalalignment='left',
                 verticalalignment='top', **ax2_params)

        fig.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=175)


if __name__ == '__main__':
    _, _, _, _, queries, if_data = dp.get_all_data(use_csv=True)
    tp_data = ifu.get_partial_scene_matches(if_data.vg_data,
                                            queries['simple_graphs'])
    iqd = ImageQueryData(queries['simple_graphs'][0], 0,
                         tp_data[0][0], if_data)
    iqd.compute_plot_data()
    iqd.generate_plot()
