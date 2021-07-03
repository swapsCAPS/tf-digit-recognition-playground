import tensorflow as tf
import numpy as np


def zoom_to_bounds(image_arr):
    mask = image_arr > 0
    coords = np.argwhere(mask)
    
    tl = (coords.min(axis=0)[0:2])
    br = (coords.max(axis=0) + 1)[0:2]
    height = br[0] - tl[0]
    width = br[1] - tl[1]
    
    new_image = tf.image.crop_to_bounding_box(
        image_arr, tl[0], tl[1], height, width
    )
    new_image = tf.image.resize(
        new_image, (28,28), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True
    )
    new_image = tf.image.resize_with_crop_or_pad(
        new_image, 28, 28
    )
    return new_image