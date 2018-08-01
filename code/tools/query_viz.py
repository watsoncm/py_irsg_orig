import os
import shutil
import json
import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.matlab.mio5_params as siom
from matplotlib import patches
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
from sklearn import mixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky

import irsg_core.data_pull as dp
import irsg_core.image_fetch_core as ifc
import irsg_core.image_fetch_utils as ifu
import irsg_core.image_fetch_querygen as ifq
import gmm_viz
from config import get_config_path

with open(get_config_path()) as f:
    cfg_data = json.load(f)
    out_path = cfg_data['file_paths']['output_path']
    data_path = cfg_data['file_paths']['mat_path']


class NotInitializedException(Exception):
    pass


class ConditionedIRSGMM(object):
    def __init__(self, gmm, rel_width, rel_height):
        self.gmm = gmm
        self.rel_width = rel_width
        self.rel_height = rel_height
        cond_data = self.condition(self.gmm, self.rel_width, self.rel_height)
        self.means, self.covariances, self.log_weights = cond_data

    # Kudos to PyPR for this conditioning algorithm
    @staticmethod
    def condition(gmm, rel_width, rel_height):
        y = np.array([rel_width, rel_height])
        new_means = []
        new_covs = []
        densities = []
        for i in range(len(gmm.means_)):
            covar_copy = copy.deepcopy(gmm.covariances_[i])
            mean_x = gmm.means_[i][:2]
            mean_y = gmm.means_[i][2:]
            A = covar_copy[:2, :2]
            B = covar_copy[2:, 2:]
            C = covar_copy[:2, 2:]
            new_mean = mean_x + np.dot(np.dot(C, np.linalg.inv(B)),
                                       (y - mean_y))
            new_cov = A - np.dot(np.dot(C, np.linalg.inv(B)), C.transpose())
            new_means.append(new_mean)
            new_covs.append(new_cov)
            densities.append(multivariate_normal.logpdf(y, mean_y, B))
        densities = np.array(densities).flatten()
        new_log_weights = (np.log(gmm.weights_) + densities -
                           logsumexp(a=densities, b=gmm.weights_))
        return new_means, new_covs, new_log_weights

    # Kudos to jakevdp for this log sampling solution, and
    # sklearn for the general sampling form
    def sample(self, n_samples=1):
        exp_samples = -np.random.exponential(size=n_samples)
        bin_probs = np.hstack([[-np.inf], self.log_weights])
        bins = np.logaddexp.accumulate(bin_probs)
        bins -= bins[-1]
        n_samples_comp = np.histogram(exp_samples, bins=bins)[0]
        samples = [np.random.multivariate_normal(mean, cov, sample)
                   for mean, cov, sample in zip(
                       self.means, self.covariances, n_samples_comp)]
        return np.vstack(samples)

    # Kudos to sklearn for the general scoring strategy
    def score_samples(self, samples):
        log_probs = [multivariate_normal.logpdf(samples, mean, cov)
                     for mean, cov in zip(self.means, self.covariances)]
        return logsumexp(np.column_stack(log_probs) + self.log_weights, axis=1)


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
        eps = np.finfo(np.float).eps
        model_pots = (model_sub_pot, model_obj_pot)
        if self.compute_gt:
            close_sub_pot = self.image_scores[self.close_sub_bbox_id,
                                              sub_pot_id]
            close_obj_pot = self.image_scores[self.close_obj_bbox_id,
                                              obj_pot_id]
            close_pots = (close_sub_pot, close_obj_pot)
            log_pots = [-np.log(pot + eps) for pot in model_pots + close_pots]
        else:
            log_pots = [-np.log(pot + eps) for pot in model_pots]
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

        eps = np.finfo(np.float).eps
        for pot_id in sub_attr_pot_ids:
            model_pot = self.image_scores[self.model_sub_bbox_id, pot_id]
            model_sub_attr_pots.append(-np.log(model_pot + eps))
            if self.compute_gt:
                gt_pot = self.image_scores[self.close_sub_bbox_id, pot_id]
                close_sub_attr_pots.append(-np.log(gt_pot + eps))
        for pot_id in obj_attr_pot_ids:
            model_pot = self.image_scores[self.model_obj_bbox_id, pot_id]
            model_obj_attr_pots.append(-np.log(model_pot + eps))
            if self.compute_gt:
                gt_pot = self.image_scores[self.close_obj_bbox_id, pot_id]
                close_obj_attr_pots.append(-np.log(gt_pot + eps))
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
    def get_gmm(means, covariances, weights):
        gmm = mixture.GaussianMixture(weights.size)
        gmm.means_ = means
        gmm.covariances_ = covariances
        gmm.covariance_type = 'full'
        gmm.weights_ = weights
        covar_data = (gmm.covariances_, gmm.covariance_type)
        gmm.precisions_cholesky_ = _compute_precision_cholesky(*covar_data)
        return gmm

    @staticmethod
    def visualize_gmm_pair(gmm, cond_gmm, rel_width, rel_height):
        x_vals, y_vals = np.linspace(-20.0, 20.0), np.linspace(-20.0, 20.0)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        w_grid = np.full(x_grid.size, rel_width)
        h_grid = np.full(x_grid.size, rel_height)

        cond_inputs = np.array([x_grid.ravel(), y_grid.ravel()]).T
        inputs = np.array([x_grid.ravel(), y_grid.ravel(), w_grid, h_grid]).T
        gmm_scores = gmm.score_samples(inputs)
        cond_gmm_scores = cond_gmm.score_samples(cond_inputs)
        z_grid = gmm_scores.reshape(x_grid.shape)
        cond_z_grid = cond_gmm_scores.reshape(x_grid.shape)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
        plt.subplots_adjust(bottom=0.2)
        ax1.set_title('Ground truth')
        ax2.set_title('Conditioned GMM')

        text_format = 'Weights: {}\nMeans: {}'
        cond_text_format = 'Log weights: {}\nMeans: {}'
        gt_text = text_format.format(gmm.weights_, gmm.means_)
        cond_text = cond_text_format.format(cond_gmm.log_weights,
                                            np.array(cond_gmm.means))
        ax1.text(0.02, -0.1, gt_text, horizontalalignment='left',
                 verticalalignment='top', transform=ax1.transAxes,
                 fontsize=10)
        ax2.text(0.02, -0.1, cond_text, horizontalalignment='left',
                 verticalalignment='top', transform=ax2.transAxes,
                 fontsize=10)

        contour = ax1.contour(x_grid, y_grid, z_grid, colors='k')
        ax1.clabel(contour, fontsize=9, inline=1)
        cond_contour = ax2.contour(x_grid, y_grid, cond_z_grid, colors='k')
        ax2.clabel(cond_contour, fontsize=9, inline=1)
        plt.show()

    def get_heatmaps(self, n_samples=250, condition_gmm=False,
                     visualize_gmm=False):
        model_sub_bbox = self.boxes[self.model_sub_bbox_id]
        model_obj_bbox = self.boxes[self.model_obj_bbox_id]
        bbox_pairs = [(model_sub_bbox, model_obj_bbox)]
        if self.compute_gt:
            bbox_pairs.append((self.image_sub_bbox, self.image_obj_bbox))

        box_maps = []
        for sub_bbox, obj_bbox in bbox_pairs:
            box_map = np.zeros((self.image.image_width,
                                self.image.image_height))
            gmm = self.get_gmm(self.pred_model.gmm_mu,
                               self.pred_model.gmm_sigma,
                               self.pred_model.gmm_weights)
            rel_width = sub_bbox[2] / obj_bbox[2]
            rel_height = sub_bbox[3] / obj_bbox[3]

            # Possibly condition and sample the GMM
            if condition_gmm:
                cond_gmm = ConditionedIRSGMM(gmm, rel_width, rel_height)
                samples = cond_gmm.sample(n_samples)
                obj_w = np.full(n_samples, obj_bbox[2])
                obj_h = np.full(n_samples, obj_bbox[3])
            else:
                cond_gmm = None
                samples = gmm.sample(n_samples)
                obj_w = samples[:, 2] * sub_bbox[2]
                obj_h = samples[:, 3] * sub_bbox[3]

            # Visualize our GMMs
            if visualize_gmm:
                self.visualize_gmm_pair(gmm, cond_gmm, rel_width, rel_height)

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

    def compute_potential_data(self, use_relationships=True):
        self.model_sub_bbox_id, self.model_obj_bbox_id = self.get_model_boxes()
        self.image_sub_bbox, self.image_obj_bbox = self.get_image_boxes()
        (self.close_sub_bbox_id, self.close_obj_bbox_id,
         self.sub_iou, self.obj_iou) = self.get_gt_closest()
        (self.model_sub_pot, self.model_obj_pot,
         self.close_sub_pot, self.close_obj_pot) = self.get_obj_scores()
        (self.model_sub_attr_pots, self.model_obj_attr_pots,
         self.close_sub_attr_pots,
         self.close_obj_attr_pots) = self.get_attr_scores()
        if use_relationships:
            self.pred_model, self.pred_model_name = self.get_pred_model()
            if self.pred_model is None:
                raise ValueError('no relevant relationship model exists')
            ((self.model_raw_pred_score, self.model_pred_score),
             (self.close_raw_pred_score, self.close_pred_score),
             (self.gt_raw_pred_score,
              self.gt_pred_score)) = self.get_pred_scores()
            (self.model_total_pot, self.gt_total_pot,
             self.close_total_pot) = self.get_total_pots()

    def compute_plot_data(self, condition_gmm=False, visualize_gmm=False):
        self.compute_potential_data()
        (self.model_heatmap,
         self.gt_heatmap) = self.get_heatmaps(condition_gmm=condition_gmm,
                                              visualize_gmm=visualize_gmm)
        self.image_array = self.get_image_array()
        self.query_text = self.get_query_text()
        self.initialized_ = True

    @staticmethod
    def get_rect(bbox, params):
        return patches.Rectangle((bbox[0], bbox[1]),
                                 bbox[2], bbox[3], **params)

    def generate_plot(self, save_path=None, sigma_blur=None):
        if not self.initialized_:
            raise NotInitializedException('ImageQueryData not initialized')

        # Set up plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 12))
        ax1.axis('off')
        ax2.axis('off')
        ax1.imshow(self.image_array, cmap='gray')
        ax2.imshow(self.image_array, cmap='gray')

        # Draw heatmaps
        if sigma_blur is not None:
            model_heatmap_blur = gaussian_filter(self.model_heatmap,
                                                 sigma=sigma_blur)
            gt_heatmap_blur = gaussian_filter(self.gt_heatmap,
                                              sigma=sigma_blur)
        else:
            model_heatmap_blur = self.model_heatmap
            gt_heatmap_blur = self.gt_heatmap
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
        ax1.text(0.02, -0.02, 'IRSG grounding'.format(self.query_text),
                 horizontalalignment='left', verticalalignment='top',
                 **ax1_params)
        ax2.text(0.02, -0.02, 'Ground truth'.format(self.query_text),
                 horizontalalignment='left', verticalalignment='top',
                 **ax2_params)
        ax1.text(0.98, -0.02, 'Query: "{}"'.format(self.query_text),
                 horizontalalignment='right', verticalalignment='top',
                 **ax1_params)
        ax2.text(0.98, -0.02, 'Query: "{}"'.format(self.query_text),
                 horizontalalignment='right', verticalalignment='top',
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
            ax2.text(0.98, 1.03, iou_text, horizontalalignment='right',
                     verticalalignment='bottom', **ax2_params)

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
        ax1.text(0.02, 1.03, model_score_text, horizontalalignment='left',
                 verticalalignment='bottom', **ax1_params)

        if self.compute_gt:
            gt_score_parts = (self.query_sub, self.close_sub_pot,
                              self.query_obj, self.close_obj_pot,
                              self.pred_model_name, self.gt_pred_score,
                              self.gt_raw_pred_score, self.close_pred_score,
                              self.close_raw_pred_score,
                              '\n'.join(gt_attr_lines), self.close_total_pot,
                              self.gt_total_pot)
            gt_score_text = gt_score_format.format(*gt_score_parts)
            ax2.text(0.02, 1.03, gt_score_text, horizontalalignment='left',
                     verticalalignment='bottom', **ax2_params)

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


def generate_all_query_plots(queries, if_data, condition_gmm=False,
                             visualize_gmm=False, negs_per_query=20):
    tp_data_pos = ifu.get_partial_scene_matches(if_data.vg_data, queries)
    tp_data_neg = generate_tp_neg(tp_data_pos, len(queries),
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
        for query_index, query in tqdm(enumerate(queries), desc='graphs',
                                       total=len(queries)):
            for image_index in tp_data[query_index]:
                image_name = 'q{:03d}_i{:03d}_{}.png'.format(query_index,
                                                             image_index,
                                                             label)
                save_path = os.path.join(output_dir, image_name)
                iqd = ImageQueryData(query, query_index, image_index, if_data,
                                     compute_gt=is_pos)
                try:
                    iqd.compute_plot_data(condition_gmm=condition_gmm,
                                          visualize_gmm=visualize_gmm)
                except ValueError:
                    print('skieeeepinging')
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
    iqd.compute_plot_data(condition_gmm=True, visualize_gmm=True)
    iqd.generate_plot()


def generate_queries_from_file(path):
    queries = []
    with open(path) as f:
        for line in f.read().splitlines():
            query_struct = siom.mat_struct()
            query_struct.annotations = ifq.gen_sro(line)
            queries.append(query_struct)
    return queries


if __name__ == '__main__':
    _, _, _, _, queries, if_data = dp.get_all_data(use_csv=True)
    path = os.path.join(data_path, 'queries.txt')
    queries = generate_queries_from_file(path)
    generate_all_query_plots(queries, if_data, condition_gmm=True,
                             visualize_gmm=False)
    # generate_test_plot(queries, if_data)
