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
