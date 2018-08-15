import re
import os
import numpy as np
from sklearn import mixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky


def get_gmm(weights, means, covariances):
    gmm = mixture.GaussianMixture(weights.size)

    # set parameters
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.covariance_type = 'full'

    # compute cholesky precisions
    covariance_data = (gmm.covariances_, gmm.covariance_type)
    gmm.precisions_cholesky_ = _compute_precision_cholesky(*covariance_data)
    return gmm


def save_gmm_data(text, path, weights, means, covariances):
    gmm_path = os.path.join(path, text)
    if not os.path.exists(gmm_path):
        os.makedirs(gmm_path)
    weights_path = os.path.join(gmm_path, 'weights.csv')
    means_path = os.path.join(gmm_path, 'means.csv')
    covar_paths = [os.path.join(gmm_path, 'covar{}.csv'.format(i))
                   for i in range(len(covariances))]
    np.savetxt(weights_path, weights)
    np.savetxt(means_path, means)
    for covar_path, covariance in zip(covar_paths, covariances):
        np.savetxt(covar_path, covariance)


def load_gmm_data(text, path, load_platt=False):
    gmm_path = os.path.join(path, text)
    weights = np.loadtxt(os.path.join(gmm_path, 'weights.csv'))
    means = np.loadtxt(os.path.join(gmm_path, 'means.csv'))
    if load_platt:
        with open(os.path.join(gmm_path, 'platt.csv')) as f:
            platt_a, platt_b = [float(v) for v in f.read().split(',')]
    covar_dict = {}
    for data_name in os.listdir(gmm_path):
        if 'covar' in data_name:
            covar_index = int(re.findall('\d+', data_name)[0])
            covar_path = os.path.join(gmm_path, data_name)
            covar_dict[covar_index] = np.loadtxt(covar_path)
    covars = np.array([covar_dict[index]
                       for index in range(len(covar_dict))])
    if load_platt:
        return weights, means, covars, platt_a, platt_b
    else:
        return weights, means, covars


def blit_sample(box_map, obj_sample):
    image_width, image_height = box_map.shape
    x, y, w, h = obj_sample.astype(int)

    # check if we're out of bounds
    out_of_bounds = False
    if w < 0 or h < 0:
        out_of_bounds = True
    if x + w <= 0 or y + h <= 0:
        out_of_bounds = True
    if x > image_width or y > image_height:
        out_of_bounds = True

    if not out_of_bounds:
        # otherwise fix partial out of bounds boxes
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if x + w > image_width - 1:
            w -= x + w - image_width
        if y + h > image_height - 1:
            h -= y + h - image_height

        # blit the final box
        box_map[x:(x + w), y:(y + h)] += np.ones((w, h))
