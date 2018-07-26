import os
import shutil
import json
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
from sklearn import mixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky

import data_pull as dp
import image_fetch_core as ifc
import image_fetch_utils as ifu
import gmm_viz

with open('config.json') as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']


# Thanks to PyPR for this implementation, which is only slightly tweaked
# from the original for compatibility with newer versions of NumPy
# and overall numerical stability.

def cond_dist(Y, centroids, ccov, mc):
    not_set_idx = np.nonzero(np.isnan(Y))[0]
    set_idx = np.nonzero(np.bitwise_not(np.isnan(Y)))[0]
    new_idx = np.concatenate((not_set_idx, set_idx))
    y = Y[set_idx]
    new_cen = []
    new_ccovs = []
    fk = []
    for i in range(len(centroids)):
        new_ccov = copy.deepcopy(ccov[i])
        new_ccov = new_ccov[:, new_idx]
        new_ccov = new_ccov[new_idx, :]
        ux = centroids[i][not_set_idx]
        uy = centroids[i][set_idx]
        A = new_ccov[:len(not_set_idx), :len(not_set_idx)]
        B = new_ccov[len(not_set_idx):, len(not_set_idx):]
        C = new_ccov[:len(not_set_idx), len(not_set_idx):]
        cen = ux + np.dot(np.dot(C, np.linalg.inv(B)), (y - uy))
        cov = A - np.dot(np.dot(C, np.linalg.inv(B)), C.transpose())
        new_cen.append(cen)
        new_ccovs.append(cov)
        fk.append(multivariate_normal.logpdf(Y[set_idx], uy, B))
    fk = np.array(fk).flatten()
    log_new_mc = np.log(mc) + fk - logsumexp(a=fk, b=mc)
    new_mc = np.exp(log_new_mc)
    return (new_cen, new_ccovs, new_mc)


class NotInitializedException(Exception):
    pass


class ImageQueryData(object):
    def __init__(self, query, query_id, image_id, if_data, compute_gt=True):
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

        self.if_data = if_data
        self.if_data.configure(self.image_id, self.query.annotations)
        potentials = self.if_data.potentials_data
        self.boxes = potentials['boxes'][self.image_id]
        self.image_scores = potentials['scores'][self.image_id]
        self.class_to_idx = potentials['class_to_idx']
        self.initialized_ = False
        self.compute_gt = compute_gt

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

        # calculate (x, y, w, h)
        bbox_a_x1 = bbox_a_x
        bbox_a_x2 = bbox_a_x + bbox_a_w
        bbox_a_y1 = bbox_a_y
        bbox_a_y2 = bbox_a_y + bbox_a_h

        bbox_b_x1 = bbox_b_x
        bbox_b_x2 = bbox_b_x + bbox_b_w
        bbox_b_y1 = bbox_b_y
        bbox_b_y2 = bbox_b_y + bbox_b_h

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
        if not self.compute_gt:
            return (None, None, None, None)
        sub_ious = [self.get_iou(box, self.image_sub_bbox)
                    for box in self.boxes]
        obj_ious = [self.get_iou(box, self.image_obj_bbox)
                    for box in self.boxes]
        sub_idx = np.argmax(sub_ious)
        obj_idx = np.argmax(obj_ious)
        return (sub_idx, obj_idx, sub_ious[sub_idx], obj_ious[obj_idx])

    def get_model_boxes(self):
        gm, _ = ifc.generate_pgm(self.if_data, verbose=False)
        self.model_energy, best_matches, _ = ifc.do_inference(gm)
        return best_matches[0], best_matches[1]

    def get_obj_scores(self):
        sub_pot_id = self.class_to_idx['obj:' + self.query_sub] - 1
        obj_pot_id = self.class_to_idx['obj:' + self.query_obj] - 1
        model_sub_pot = self.image_scores[self.model_sub_bbox_id, sub_pot_id]
        model_obj_pot = self.image_scores[self.model_obj_bbox_id, obj_pot_id]
        if self.compute_gt:
            close_sub_pot = self.image_scores[self.close_sub_bbox_id,
                                              sub_pot_id]
            close_obj_pot = self.image_scores[self.close_obj_bbox_id,
                                              obj_pot_id]
            log_pots = [-np.log(pot) for pot in (model_sub_pot, model_obj_pot,
                                                 close_sub_pot, close_obj_pot)]
        else:
            log_pots = [-np.log(pot) for pot in (model_sub_pot, model_obj_pot)]
            log_pots.extend([None, None])
        return log_pots

    def get_attr_scores(self):
        sub_attr_pot_ids = [self.class_to_idx['atr:' + attr] - 1
                            for attr in self.query_sub_attrs]
        obj_attr_pot_ids = [self.class_to_idx['atr:' + attr] - 1
                            for attr in self.query_obj_attrs]
        model_sub_attr_pots, model_obj_attr_pots = [], []
        if self.compute_gt:
            close_sub_attr_pots, close_obj_attr_pots = [], []
        else:
            close_sub_attr_pots, close_obj_attr_pots = None, None

        for pot_id in sub_attr_pot_ids:
            model_pot = self.image_scores[self.model_sub_bbox_id, pot_id]
            model_sub_attr_pots.append(-np.log(model_pot))
            if self.compute_gt:
                gt_pot = self.image_scores[self.close_sub_bbox_id, pot_id]
                close_sub_attr_pots.append(-np.log(gt_pot))
        for pot_id in obj_attr_pot_ids:
            model_pot = self.image_scores[self.model_obj_bbox_id, pot_id]
            model_obj_attr_pots.append(-np.log(model_pot))
            if self.compute_gt:
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
        log_scores = np.log(eps + scores)
        sig_scores = 1.0 / (1.0 + np.exp(self.pred_model.platt_a * log_scores +
                                         self.pred_model.platt_b))
        return scores[0], -np.log(sig_scores[0])

    def get_pred_scores(self):
        model_sub_bbox = self.boxes[self.model_sub_bbox_id]
        model_obj_bbox = self.boxes[self.model_obj_bbox_id]
        model_pred_scores = self.score_box_pair(model_sub_bbox, model_obj_bbox)
        if self.compute_gt:
            close_sub_bbox = self.boxes[self.close_sub_bbox_id]
            close_obj_bbox = self.boxes[self.close_obj_bbox_id]
            close_pred_scores = self.score_box_pair(close_sub_bbox,
                                                    close_obj_bbox)
            gt_pred_scores = self.score_box_pair(self.image_sub_bbox,
                                                 self.image_obj_bbox)
        else:
            close_pred_scores, gt_pred_scores = (None, None), (None, None)
        return model_pred_scores, close_pred_scores, gt_pred_scores

    @staticmethod
    def get_conditional_gmm(pred_model, width, height):
        print((pred_model.gmm_mu, pred_model.gmm_sigma,
               pred_model.gmm_weights))
        result = cond_dist(np.array([np.nan, np.nan, width, height]),
                           list(pred_model.gmm_mu), list(pred_model.gmm_sigma),
                           list(pred_model.gmm_weights))
        new_mu, new_sigma, new_weights = [np.array(part) for part in result]
        gmm = mixture.GaussianMixture(new_weights.size)
        gmm.means_ = new_mu
        gmm.covariances_ = new_sigma
        gmm.covariance_type = 'full'
        gmm.weights_ = new_weights

        covar_data = (gmm.covariances_, gmm.covariance_type)
        gmm.precisions_cholesky_ = _compute_precision_cholesky(*covar_data)

        gmm2 = mixture.GaussianMixture(pred_model.gmm_weights.size)
        gmm2.means_ = pred_model.gmm_mu
        gmm2.covariances_ = pred_model.gmm_sigma
        gmm2.covariance_type = 'full'
        gmm2.weights_ = pred_model.gmm_weights

        covar_data2 = (gmm2.covariances_, gmm2.covariance_type)
        gmm2.precisions_cholesky_ = _compute_precision_cholesky(*covar_data2)
        import pdb; pdb.set_trace()

        return gmm

    def get_heatmaps(self, n_samples=250):
        model_sub_bbox = self.boxes[self.model_sub_bbox_id]
        model_obj_bbox = self.boxes[self.model_obj_bbox_id]
        bbox_pairs = [(model_sub_bbox, model_obj_bbox)]
        if self.compute_gt:
            bbox_pairs.append((self.image_sub_bbox, self.image_obj_bbox))

        box_maps = []
        for sub_bbox, obj_bbox in bbox_pairs:
            # Condition and sample the GMM
            rel_width = sub_bbox[2] / obj_bbox[2]
            rel_height = sub_bbox[3] / obj_bbox[3]
            gmm = self.get_conditional_gmm(self.pred_model, rel_width,
                                           rel_height)
            samples = gmm.sample(n_samples)[0]

            # Figure out width and height
            box_map = np.zeros((self.image.image_width,
                                self.image.image_height))
            # Calculate center x and y
            sub_bbox_center_x = (sub_bbox[0] + 0.5 * sub_bbox[2])
            sub_bbox_center_y = (sub_bbox[1] + 0.5 * sub_bbox[3])
            obj_w, obj_h = np.full(250, obj_bbox[2]), np.full(250, obj_bbox[3])
            obj_x = (sub_bbox_center_x - sub_bbox[2] * samples[:, 0] -
                     0.5 * obj_w)
            obj_y = (sub_bbox_center_y - sub_bbox[3] * samples[:, 1] -
                     0.5 * obj_h)

            # Blit all samples
            obj_samples = np.vstack((obj_x, obj_y, obj_w, obj_h)).T
            for obj_sample in obj_samples:
                print(obj_sample)
                gmm_viz.blit_sample(box_map, obj_sample)
            box_maps.append(box_map)

        if self.compute_gt:
            model_heatmap, gt_heatmap = box_maps
        else:
            model_heatmap, gt_heatmap = box_maps[0], None
        return model_heatmap, gt_heatmap

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

        if self.compute_gt:
            gt_total_pot = (self.close_sub_pot + self.close_obj_pot +
                            self.gt_pred_score)
            close_total_pot = (self.close_sub_pot + self.close_obj_pot +
                               self.close_pred_score)
            close_addr_total = sum(self.close_sub_attr_pots +
                                   self.close_obj_attr_pots)
            gt_total_pot += close_addr_total
            close_total_pot += close_addr_total
        else:
            close_total_pot, gt_total_pot = None, None

        return model_total_pot, close_total_pot, gt_total_pot

    def get_image_boxes(self):
        if not self.compute_gt:
            return None, None
        unary_triples = self.make_array(self.image.annotations.unary_triples)
        binary_triples = self.make_array(self.image.annotations.binary_triples)
        model_sub_bbox = self.boxes[self.model_sub_bbox_id]
        model_obj_bbox = self.boxes[self.model_obj_bbox_id]
        image_objects = self.image.annotations.objects
        bbox_pairs = []
        mean_ious = []
        for triple in binary_triples:
            trip_sub, trip_pred, trip_obj = triple.text
            sub_attrs, obj_attrs = self.get_attributes(unary_triples,
                                                       triple.subject,
                                                       triple.object)
            if (self.triple_matches(trip_sub, trip_pred, trip_obj,
                                    sub_attrs, obj_attrs)):
                sub_bbox = self.make_bbox(image_objects[triple.subject].bbox)
                obj_bbox = self.make_bbox(image_objects[triple.object].bbox)
                sub_iou = self.get_iou(sub_bbox, model_sub_bbox)
                obj_iou = self.get_iou(obj_bbox, model_obj_bbox)
                bbox_pairs.append((sub_bbox, obj_bbox))
                mean_ious.append(np.mean((sub_iou, obj_iou)))
        if len(bbox_pairs) == 0:
            raise ValueError('image incompatible with query')
        bbox_index = np.argmax(mean_ious)
        return bbox_pairs[bbox_index]

    def compute_plot_data(self):
        self.model_sub_bbox_id, self.model_obj_bbox_id = self.get_model_boxes()
        self.image_sub_bbox, self.image_obj_bbox = self.get_image_boxes()
        (self.close_sub_bbox_id, self.close_obj_bbox_id,
         self.sub_iou, self.obj_iou) = self.get_gt_closest()
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
        heatmaps = ((model_heatmap_blur, gt_heatmap_blur) if self.compute_gt
                    else (model_heatmap_blur,))
        vmin, vmax = np.min(heatmaps), np.max(heatmaps)
        ax1.imshow(model_heatmap_blur.T, vmin=vmin,
                   vmax=vmax, alpha=0.4)
        if self.compute_gt:
            ax2.imshow(gt_heatmap_blur.T, vmin=vmin,
                       vmax=vmax, alpha=0.4)

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

        if self.compute_gt:
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
        if self.compute_gt:
            for attr, gt_pot in zip(self.query_sub_attrs,
                                    self.close_sub_attr_pots):
                gt_line = attr_format.format(attr, self.query_sub, gt_pot)
                gt_attr_lines.append(gt_line)
            for attr, gt_pot in zip(self.query_obj_attrs,
                                    self.close_obj_attr_pots):
                gt_line = attr_format.format(attr, self.query_obj, gt_pot)
                gt_attr_lines.append(gt_line)

        for attr, model_pot in zip(self.query_sub_attrs,
                                   self.model_sub_attr_pots):
            model_line = attr_format.format(attr, self.query_sub, model_pot)
            model_attr_lines.append(model_line)
        for attr, model_pot in zip(self.query_obj_attrs,
                                   self.model_obj_attr_pots):
            model_line = attr_format.format(attr, self.query_obj, model_pot)
            model_attr_lines.append(model_line)

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
        model_score_text = model_score_format.format(*model_score_parts)
        ax1.text(0.02, 0.98, model_score_text, horizontalalignment='left',
                 verticalalignment='top', **ax1_params)

        if self.compute_gt:
            gt_score_parts = (self.query_sub, self.close_sub_pot,
                              self.query_obj, self.close_obj_pot,
                              self.pred_model_name, self.gt_pred_score,
                              self.gt_raw_pred_score, self.close_pred_score,
                              self.close_raw_pred_score,
                              '\n'.join(gt_attr_lines), self.close_total_pot,
                              self.gt_total_pot)
            gt_score_text = gt_score_format.format(*gt_score_parts)
            ax2.text(0.02, 0.98, gt_score_text, horizontalalignment='left',
                     verticalalignment='top', **ax2_params)

        fig.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=175)
        plt.close()


def generate_tp_neg(tp_data_pos, n_queries, n_images, negs_per_query):
    tp_data_neg = []
    for query_index in range(n_queries):
        image_indexes = [index for index in range(n_images)
                         if index not in tp_data_pos[query_index]]
        tp_data_neg.append(np.random.choice(image_indexes,
                                            size=negs_per_query))
    return tp_data_neg


def generate_all_query_plots(queries, if_data, negs_per_query=20):
    simple_graphs = queries['simple_graphs']
    tp_data_pos = ifu.get_partial_scene_matches(if_data.vg_data, simple_graphs)
    tp_data_neg = generate_tp_neg(tp_data_pos, len(simple_graphs),
                                  len(if_data.vg_data), negs_per_query)
    output_dir = os.path.join(out_path, 'query_viz')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    positive_energies = {}
    negative_energies = {}

    # Run all positives and negatives
    for tp_data, energies, is_pos in ((tp_data_pos, positive_energies, True),
                                      (tp_data_neg, negative_energies, False)):
        label = 'pos' if is_pos else 'neg'
        for query_index, query in tqdm(enumerate(simple_graphs), desc='graphs',
                                       total=len(simple_graphs)):
            for image_index in tp_data[query_index]:
                image_name = 'q{:03d}_i{:03d}_{}.png'.format(query_index,
                                                             image_index,
                                                             label)
                save_path = os.path.join(output_dir, image_name)
                iqd = ImageQueryData(query, query_index, image_index, if_data,
                                     compute_gt=is_pos)
                try:
                    iqd.compute_plot_data()
                except ValueError:
                    continue
                iqd.generate_plot(save_path=save_path)
                energies[save_path] = iqd.model_energy

    # Copy over all relevant images
    top_pos_dir = os.path.join(out_path, 'query_top_pos')
    top_neg_dir = os.path.join(out_path, 'query_top_neg')
    if not os.path.exists(top_pos_dir):
        os.mkdir(top_pos_dir)
    if not os.path.exists(top_neg_dir):
        os.mkdir(top_neg_dir)
    pos_pairs = sorted(positive_energies.items(), key=lambda pair: pair[1])
    neg_pairs = sorted(negative_energies.items(), key=lambda pair: pair[1])
    for path, _ in pos_pairs[:10]:
        shutil.copy(path, top_pos_dir)
    for path, _ in neg_pairs[:10]:
        shutil.copy(path, top_neg_dir)


def generate_test_plot(queries, if_data):
    iqd = ImageQueryData(queries['simple_graphs'][4], 4, 38, if_data)
    iqd.compute_plot_data()
    iqd.generate_plot()


if __name__ == '__main__':
    _, _, _, _, queries, if_data = dp.get_all_data(use_csv=True)
    # generate_all_query_plots(queries, if_data)
    generate_test_plot(queries, if_data)
