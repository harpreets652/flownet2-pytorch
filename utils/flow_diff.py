import numpy as np
import cv2
import skimage.measure as sk_image
import utils.flow_utils as fu
import sys


def flow_difference(flow_a, flow_b, patch_size, threshold=False, difference_func="absolute"):
    """
    Compute pixel-level difference between source and target

    :param flow_a: 2D array, target flow
    :param flow_b: 2D array, flow to be evaluated
    :param patch_size: optional, pixel value of flow_b is an average in patch size; odd number size plz
    :param threshold: optional, if difference should be thresholded
    :param difference_func: absolute|squared
    :return: x_diff, y_diff numpy arrays with range between [0, 1]
    """

    this_module = sys.modules[__name__]
    if not hasattr(this_module, difference_func + "_difference"):
        raise RuntimeError(f"Difference function {difference_func}_difference not found")

    diff_func = getattr(this_module, difference_func + "_difference")

    diff_y = _flow_diff(flow_a[:, :, 0], flow_b[:, :, 0], patch_size, threshold, diff_func)
    diff_x = _flow_diff(flow_a[:, :, 1], flow_b[:, :, 1], patch_size, threshold, diff_func)

    return diff_x, diff_y


def flow_difference_sk_image(flow_a, flow_b):
    flow_img_a = fu.flow_2_rgb(flow_a)
    flow_img_b = fu.flow_2_rgb(flow_b)

    # note~ have to convert image to gray
    flow_img_a = cv2.cvtColor(flow_img_a, cv2.COLOR_RGB2GRAY)
    flow_img_b = cv2.cvtColor(flow_img_b, cv2.COLOR_RGB2GRAY)

    (mean_ssim, diff_image) = sk_image.compare_ssim(flow_img_a, flow_img_b, full=True)
    diff_image = diff_image * 255

    # note~ I can also threshold in order to reduce noise
    # thresh = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    print(f"DEBUG: mean ssim: {mean_ssim}")

    return diff_image


def _flow_diff(flow_source, flow_test, patch_size, threshold, difference_func):
    """
    difference between source and test
    :param flow_source: 1D array, source flow of a single axis
    :param flow_test: 1D array, flow to evaluate
    :param patch_size:
    :param patch_size: optional, pixel value of flow_b is an average in patch size; odd number size plz
    :param threshold: optional, if thresholding should be applied
    :param difference_func: function object
    :return: result
    """

    """
    diff image using selected difference function, absolute difference, squared difference
    if patch provided, use patch to average error
    if tolerance provided, apply tolerance; exact values will be near 0.
    scale difference to be between 0...1
    """

    difference = difference_func(flow_source, flow_test)

    if patch_size:
        # for custom kernel: cv2.filter2D
        pad_size = (patch_size[0] // 2, patch_size[1] // 2)
        difference = np.pad(difference, pad_size, 'constant', constant_values=0)
        difference = cv2.boxFilter(difference, -1, patch_size, normalize=True)

    if threshold:
        difference = cv2.threshold(difference, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        difference = cv2.normalize(difference, None, 0, 1, norm_type=cv2.NORM_MINMAX)

    return difference


def squared_difference(a, b):
    """
    (a - b)^2

    :param a: ndarray
    :param b: ndarray
    :return: result
    """
    return np.square(np.subtract(a, b))


def absolute_difference(a, b):
    """
    abs(a - b)

    :param a: ndarray
    :param b: ndarray
    :return: result
    """
    return np.abs(np.subtract(a, b))
