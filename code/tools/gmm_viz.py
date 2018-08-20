import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

import gmm_utils
import irsg_core.data_pull as dp


def plot_heatmap(box_map, sub_bbox, obj_bbox, image_index, sub_index,
                 obj_index, model_name, vg_data, ifdata, save_path=None):
    """Plots a simple heatmap given an array of heat values, bounding
    boxes for subject and object, the index of the image, etc."""
    fig, ax = plt.subplots(1)
    plt.xticks([])
    plt.yticks([])

    # plot boxes around subject and object
    sub_patch = patches.Rectangle((sub_bbox.x, sub_bbox.y), sub_bbox.w,
                                  sub_bbox.h, linewidth=3, edgecolor='red',
                                  facecolor='none')
    obj_patch = patches.Rectangle((obj_bbox.x, obj_bbox.y), obj_bbox.w,
                                  obj_bbox.h, linewidth=3, edgecolor='green',
                                  facecolor='none')
    ax.add_patch(sub_patch)
    ax.add_patch(obj_patch)

    # load image
    image_name = os.path.basename(vg_data[image_index].image_path)
    image_path = os.path.join(ifdata.base_image_path, image_name)
    pil_image = Image.open(image_path).convert("L")
    image_array = np.array(pil_image)

    # plot the box map and the image
    plt.imshow(image_array, cmap='gray')
    box_map_blur = gaussian_filter(box_map, sigma=7)
    plt.imshow(box_map_blur.T, alpha=0.4)
    plt.tight_layout()

    # set the title
    image_objs = vg_data[image_index].annotations.objects
    sub_name = np.array(image_objs[sub_index].names).reshape(-1)[0]
    obj_name = np.array(image_objs[obj_index].names).reshape(-1)[0]
    plt.title('{} ({} -> {})'.format(model_name, sub_name, obj_name))
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=175)
    plt.close()


def get_heatmap(gmm, image_index, sub_index, obj_index, vg_data,
                n_samples=1000):
    """Generates a heatmap from a GMM and a set of indices."""
    image_data = vg_data[image_index]
    samples = gmm.sample(n_samples)[0]
    image_width, image_height = image_data.image_width, image_data.image_height
    box_map = np.zeros((image_width, image_height))

    # figure out width and height for object
    sub_bbox = image_data.annotations.objects[sub_index].bbox
    obj_bbox = image_data.annotations.objects[obj_index].bbox
    obj_w = samples[:, 2] * sub_bbox.w
    obj_h = samples[:, 3] * sub_bbox.h

    # calculate center x and y
    sub_bbox_center_x = (sub_bbox.x + 0.5 * sub_bbox.w)
    sub_bbox_center_y = (sub_bbox.y + 0.5 * sub_bbox.h)
    obj_x = sub_bbox_center_x - sub_bbox.w * samples[:, 0] - 0.5 * obj_w
    obj_y = sub_bbox_center_y - sub_bbox.h * samples[:, 1] - 0.5 * obj_h

    # blit all samples
    obj_samples = np.vstack((obj_x, obj_y, obj_w, obj_h)).T
    for obj_sample in obj_samples:
        gmm_utils.blit_sample(box_map, obj_sample)
    return box_map, sub_bbox, obj_bbox


def visualize_gmm(model_name, image_index, sub_index, obj_index, vg_data,
                  ifdata, save_path=None):
    """Fully visualizes a GMM given a model name and a series of indices."""
    gmm_params = ifdata.relationship_models[model_name]
    gmm = gmm_utils.get_gmm(gmm_params.gmm_weights,
                            gmm_params.gmm_mu, gmm_params.gmm_sigma)
    box_map, sub_bbox, obj_bbox = get_heatmap(gmm, image_index, sub_index,
                                              obj_index, vg_data)
    plot_heatmap(box_map, sub_bbox, obj_bbox, image_index, sub_index,
                 obj_index, model_name, vg_data, ifdata, save_path=save_path)


if __name__ == '__main__':
    vgd, _, _, _, _, ifdata = dp.get_all_data(use_csv=True)
    visualize_gmm('man_beside_man', 421, 0, 10, vgd['vg_data_test'],
                  ifdata)
