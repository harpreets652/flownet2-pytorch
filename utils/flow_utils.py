import numpy as np
import cv2
import sys
import skimage.measure as sk_image

TAG_CHAR = np.array([202021.25], np.float32)


def read_flow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def write_flow(filename, uv, v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def flow_to_image_hsv(flow):
    """
    Flow visualization example from OpenCV Python example
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def compute_optical_flow_farneback(frame_a, frame_b, include_magnitude=True):
    """
    Computes optical flow between two frames, assuming images are grayscale

    :param frame_a: frame N
    :param frame_b: frame N+1
    :param include_magnitude: append magnitude to optical flow array
    :return: optical flow + magnitude
    """

    flow = cv2.calcOpticalFlowFarneback(frame_a, frame_b, None, 0.5, 3, 9, 6, 7, 1.5, 0)

    if include_magnitude:
        mag, _ = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        mag = np.expand_dims(mag, -1)
        flow = np.dstack((flow, mag))

    return flow


def show_flow(flow_file, win_name='', wait_time=0):
    """
    Show optical flow where flow is a file name

    Args:
        flow_file(str): optical flow to be shown
        win_name(str): the window name
        wait_time(int): value of waitKey param
    """
    flow = read_flow(flow_file)
    flow_img = flow_2_rgb(flow)
    # flow_img = flow_to_image_hsv(flow)
    cv2.imshow(win_name, flow_img)
    cv2.waitKey(wait_time)
    return


def display_flow(flow):
    """
    Show optical flow
    :param flow:  np array
    """

    flow_img = flow_2_rgb(flow)
    # flow_img = flow_to_image_hsv(flow)
    cv2.imshow("Flow", flow_img)
    cv2.waitKey(0)
    return


def display_results(input_flow, input_images, output_flow):
    """
    Display inference results

    :param input_flow: target flow
    :param input_images: input images
    :param output_flow: network output
    """
    image_a = input_images[0]
    image_b = input_images[1]
    rgb_frames = np.hstack((image_a, image_b))

    input_flow_image = flow_2_rgb(input_flow)
    output_flow_image = flow_2_rgb(output_flow)
    flow_viz_frames = np.hstack((input_flow_image, output_flow_image))

    x_diff, y_diff = flow_difference(input_flow, output_flow, (5, 5), difference_func="squared")
    flow_diffs = np.hstack((x_diff, y_diff))

    cv2.imshow("image a, image b", rgb_frames)
    cv2.imshow("target flow, generated flow", flow_viz_frames)
    cv2.imshow("x diff, y diff", flow_diffs)
    cv2.waitKey(0)
    return


def flow_2_rgb(flow, color_wheel=None, unknown_thr=1e6):
    """Convert flow map to RGB image

    Args:
        flow(ndarray): optical flow
        color_wheel(ndarray or None): color wheel used to map flow field to RGB
            colorspace. Default color wheel will be used if not specified
        unknown_thr(str): values above this threshold will be marked as unknown
            and thus ignored

    Returns:
        ndarray: an RGB image that can be visualized
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
                   (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)
        dx /= max_rad
        dy /= max_rad

    rad = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(-dy, -dx) / np.pi

    bin_real = (angle + 1) / 2 * (num_bins - 1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    flow_img = (1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]
    small_ind = rad <= 1
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img


def make_color_wheel(bins=None):
    """Build a color wheel

    Args:
        bins(list or tuple, optional): specify number of bins for each color
            range, corresponding to six ranges: red -> yellow, yellow -> green,
            green -> cyan, cyan -> blue, blue -> magenta, magenta -> red.
            [15, 6, 4, 11, 13, 6] is used for default (see Middlebury).

    Returns:
        ndarray: color wheel of shape (total_bins, 3)
    """
    if bins is None:
        bins = [15, 6, 4, 11, 13, 6]
    assert len(bins) == 6

    RY, YG, GC, CB, BM, MR = tuple(bins)

    ry = [1, np.arange(RY) / RY, 0]
    yg = [1 - np.arange(YG) / YG, 1, 0]
    gc = [0, 1, np.arange(GC) / GC]
    cb = [0, 1 - np.arange(CB) / CB, 1]
    bm = [np.arange(BM) / BM, 0, 1]
    mr = [1, 0, 1 - np.arange(MR) / MR]

    num_bins = RY + YG + GC + CB + BM + MR

    color_wheel = np.zeros((3, num_bins), dtype=np.float32)

    col = 0
    for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
        for j in range(3):
            color_wheel[j, col:col + bins[i]] = color[j]
        col += bins[i]

    return color_wheel.T


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

    diff_x = _flow_diff(flow_a[:, :, 0], flow_b[:, :, 0], patch_size, threshold, diff_func)
    diff_y = _flow_diff(flow_a[:, :, 1], flow_b[:, :, 1], patch_size, threshold, diff_func)

    return diff_x, diff_y


def flow_difference_sk_image(flow_a, flow_b):
    flow_img_a = flow_2_rgb(flow_a)
    flow_img_b = flow_2_rgb(flow_b)

    # note~ have to convert image to gray
    flow_img_a = cv2.cvtColor(flow_img_a, cv2.COLOR_RGB2GRAY)
    flow_img_b = cv2.cvtColor(flow_img_b, cv2.COLOR_RGB2GRAY)

    (mean_ssim, diff_image) = sk_image.compare_ssim(flow_img_a, flow_img_b, full=True)
    diff_image = np.clip(diff_image * 255, 0, 255)

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

    difference = difference_func(flow_source, flow_test)

    if patch_size:
        # for custom kernel: cv2.filter2D
        pad_size = (patch_size[0] // 2, patch_size[1] // 2)
        difference = np.pad(difference, pad_size, 'constant', constant_values=0)
        difference = cv2.boxFilter(difference, -1, patch_size, normalize=True)

        # remove padding
        diff_shape = difference.shape
        crop_rows = (pad_size[0], diff_shape[0] - pad_size[0])
        crop_cols = (pad_size[1], diff_shape[1] - pad_size[1])
        difference = difference[crop_rows[0]:crop_rows[1], crop_cols[0]:crop_cols[1]]

    difference = cv2.normalize(difference, None, 0, 1, norm_type=cv2.NORM_MINMAX)
    if threshold:
        # fixme: if global error is low, this will find relative peaks; how to think about threshold as percent tolerance?
        difference = (difference * 255).astype(np.uint8)
        difference = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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
