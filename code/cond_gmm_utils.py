import tensorflow as tf
from tensorflow import keras


def get_cond_gmm_model():
    sub_box = keras.Input(shape=(5,), name='subject_box')
    obj_box = keras.Input(shape=(5,), name='subject_box')
    vgg = keras.applications.VGG16
