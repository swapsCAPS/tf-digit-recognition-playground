import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import math
from typing import Tuple, Union
from deskew import determine_skew
import imutils

# Takes a grayscale image
def black_blur(image: np.ndarray, blur: int = 13, threshold: int = 250):
    # TODO what if it is gray already? or other shape?
    print(image.shape)

    res = None

    if len(image.shape) > 2:
        res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    th, res = cv2.threshold(res, threshold, 255, cv2.THRESH_BINARY)
    if blur > 0:
        res = cv2.GaussianBlur(res, (blur,blur), 0)
    return res

def deskew(image: np.ndarray, blur: int = 13, threshold: int = 250, debug=False, do_blur=True):
    def rotate(
            image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
    ) -> np.ndarray:
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

    image_copy = image.copy()

    if do_blur:
        image_copy = black_blur(image_copy, blur, threshold)

    angle = determine_skew(image_copy)

    if (debug is True): print(f"deskew() Found angle: {angle}")

    if (angle is not None):
        if (angle < -75): angle = angle + 90
        image_copy = rotate(image_copy, angle, (255, 255, 255))

    return image_copy

def crop_to_content(image_arr):
    mask = image_arr > 0
    coords = np.argwhere(mask)

    tl = (coords.min(axis=0)[0:2])
    br = (coords.max(axis=0) + 1)[0:2]
    height = br[0] - tl[0]
    width = br[1] - tl[1]

    return image_arr[tl[0]:tl[0] + height, tl[1]:tl[1] + width]

def zoom_to_bounds(image_arr):
    new_image = crop_to_content(image_arr)

    new_image = tf.image.resize(
        new_image, (28,28), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True
    )
    new_image = tf.image.resize_with_crop_or_pad(
        new_image, 28, 28
    )
    return new_image

# Expects image and template both with alpha channel
def simpleMatchTemplate(
    img: np.ndarray,
    tmpl: np.ndarray,
    threshold=1,
    norm=False,
    debug=False,
    rect_thresh = 1.5,
):
    def print_debug(*args):
        if (not debug): return
        print("simpleMatchTemplate()", *args)

    if img is None:
        raise Exception("Please provide an image")

    if tmpl is None:
        raise Exception("Please provide an template")

    print_debug(f"img.shape:        {img.shape}")
    print_debug(f"tmpl.shape:       {tmpl.shape}")

    if img.shape[2] != 4:
        raise Exception("Image needs alpha channel")
    if tmpl.shape[2] != 4:
        raise Exception("Template needs alpha channel")

    alpha_mask = np.array(cv2.split(tmpl)[3])
    print('alpha_mask', alpha_mask)
    cropped = np.argwhere(alpha_mask < 255)
    print('cropped', cropped)

    print_debug(f"alpha_mask.shape: {alpha_mask.shape}")
    print_debug(f"cropped.shape:    {cropped.shape}")

    res = cv2.matchTemplate(img, tmpl, cv2.TM_CCORR_NORMED, mask=alpha_mask)

    print_debug('res', res)

    #  if (debug):
        #  plt.figure(figsize=(25, 25))
        #  plt.imshow(res)
        #  plt.show()

    # Normalize from 0 to 1
    if norm:
        print_debug(f"normalizing")
        res = cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX, -1)

    match_locations = np.where(res >= threshold)

    print_debug('match_locations', match_locations)

    h, w = tmpl.shape[:2]

    rects = []
    for (x, y) in zip(*match_locations[::-1]):
        r = [ int(x), int(y), w, h ]
        rects.append(r)
        # lol do it twice...
        rects.append(r)

    rects, weights = cv2.groupRectangles(rects, 1, rect_thresh)

    return rects

###
# Takes an image, column, row and amount of rows to run for
# Goes through each row and tries to find the first non 0 pixel at given column
# TODO is there a more opencv-ic way? Numpy magic? find first on axis or smth?
#      this will be quite slow ; )
###
def find_white_pixel(image, column, from_row, row_amount):
    found = None
    for i, r in enumerate(image[from_row:from_row + row_amount]):
        px = r[column]
        if (px > 0):
            found = (column, from_row + i)
            break
    if (found is None):
        raise Exception("Did not find pixel")
    return found

def disp_image(image):
    plt.figure(figsize=(25, 25))
    plt.imshow(image)
    plt.show()

def findHoriLines(image, max_height=100, min_width=1000, debug=False):
    # these are coords yay!
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    areas = [cv2.contourArea(c) for c in cnts]
    polys = [cv2.approxPolyDP(c, 3, True) for c in cnts]
    rects = [cv2.boundingRect(p) for p in polys]

    result = []
    for r in rects:
        x, y, w, h = r

        if abs(h) > max_height: continue
        if abs(w) < min_width: continue

        result.append(r)

    # Sort top to bottom
    result.sort(key=lambda r: r[1])

    def draw():
        copy = image.copy()
        copy = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
        cv2.drawContours(copy, cnts, -1, (0, 255, 255), 2)

        for x, y, w, h in result:
            cv2.rectangle(copy, (x, y), (x + w, y + h), (255, 0, 0), 5)
        disp_image(copy)

    if (debug is True):
        draw()

    if len(result) != 2:
        draw()
        raise Exception(f"findHoriLines() found only [{len(result)}] line(s)")

    return result


def straighten_front(
    orig,
    pad=150,
    border_value=(255, 255, 255),
    debug=False
):
    img_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    th, thresh = cv2.threshold(img_gray, 254, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=5)
    thresh = 255 - thresh

    [ first, last ] = findHoriLines(thresh, min_width=thresh.shape[1] - 300, debug=False)

    tl = find_white_pixel(thresh, first[0]               , first[1], first[3])
    tr = find_white_pixel(thresh, first[0] + first[2] - 1, first[1], first[3])
    bl = find_white_pixel(thresh, last[0]                , last[1],  last[3])
    br = find_white_pixel(thresh, last[0]  + last[2]  - 1, last[1],  last[3])

    actual = np.float32([tl, tr, bl, br])
    rows, cols, d = orig.shape
    wanted = np.float32([ [pad, pad], [cols - pad, pad], [pad, rows - pad], [cols - pad, rows - pad] ])
    M, mask = cv2.findHomography(actual,wanted)
    dst = cv2.warpPerspective(orig, M, (cols, rows), borderValue=border_value)

    if (debug is True): disp_image(dst)
    return dst
