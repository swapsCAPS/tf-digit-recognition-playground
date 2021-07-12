import tensorflow as tf
import numpy as np
import cv2

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

# Expects image and template both with alpha channel
def simpleMatchTemplate(img, tmpl, threshold=1, norm=False):    
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    tmpl = cv2.imread(tmpl, cv2.IMREAD_UNCHANGED)

    if img.shape[2] != 4:
        raise Exception("Image needs alpha channel")
    if tmpl.shape[2] != 4:
        raise Exception("Template needs alpha channel")
        
    alpha_mask = np.array(cv2.split(tmpl)[3])
    
    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCORR_NORMED, mask=alpha_mask)
    # Normalize from 0 to 1
    if norm:
        res = cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX, -1)
    
    match_locations = np.where(res >= threshold)
    
    h, w = tmpl.shape[:2]
    
    r = []
    for (x, y) in zip(*match_locations[::-1]):
        r.append(((x, y), (x + w, y + h)))
        
    return r