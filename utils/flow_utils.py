import numpy as np
import cv2
import sys

TAG_CHAR = np.array([202021.25], np.float32)
LABEL_ANOMALOUS = 0
LABEL_NORMAL = 1


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

    x_diff, y_diff = flow_difference(input_flow, output_flow, (5, 5), difference_func="absolute", use_mag_ang=False)
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


def acute_angle_diff_filter(angle):
    return angle if angle <= 180.0 else 360.0 - angle


vectorized_diff_filter = np.vectorize(acute_angle_diff_filter)


def flow_difference(flow_a, flow_b, patch_size, use_mag_ang=False, threshold=False, difference_func="absolute"):
    """
    Compute pixel-level difference between source and target

    :param flow_a: 2D array, target flow
    :param flow_b: 2D array, flow to be evaluated
    :param patch_size: optional, pixel value of flow_b is an average in patch size; odd number size plz
    :param use_mag_ang: find difference in magnitude/angle rather than raw flow vectors
    :param threshold: optional, if difference should be thresholded
    :param difference_func: absolute|squared
    :return: x_diff, y_diff numpy arrays with range between [0, 1]
    """

    this_module = sys.modules[__name__]

    if use_mag_ang and difference_func != "absolute":
        print("Using mag/angle comparision, defaulting to absolute difference")
        difference_func = "absolute"

    if not hasattr(this_module, difference_func + "_difference"):
        raise RuntimeError(f"Difference function {difference_func}_difference not found")

    diff_func = getattr(this_module, difference_func + "_difference")

    if use_mag_ang:
        # magnitude and angle(degrees)
        flow_source = np.dstack(cv2.cartToPolar(flow_a[:, :, 0], flow_a[:, :, 1], angleInDegrees=True))
        flow_target = np.dstack(cv2.cartToPolar(flow_b[:, :, 0], flow_b[:, :, 1], angleInDegrees=True))

        # magnitude difference
        diff_chan_0 = diff_func(flow_source[:, :, 0], flow_target[:, :, 0])

        # angle difference
        angle_diffs = diff_func(flow_source[:, :, 1], flow_target[:, :, 1])

        # find acute angle between differences
        diff_chan_1 = vectorized_diff_filter(angle_diffs)

        # max magnitude difference is (25, 25) = 35.35
        # max angle max difference is 180.0
        # note: opposite angles will have 0 mag difference but max angle difference
        chan_0_norm = 35.36
        chan_1_norm = 180.0
    else:
        diff_chan_0 = diff_func(flow_a[:, :, 0], flow_b[:, :, 0])
        diff_chan_1 = diff_func(flow_a[:, :, 1], flow_b[:, :, 1])

        # max absolute difference will be 50
        chan_0_norm, chan_1_norm = 50.0, 50.0

    diff_chan_0 = _flow_diff(diff_chan_0, patch_size, threshold, chan_0_norm)
    diff_chan_1 = _flow_diff(diff_chan_1, patch_size, threshold, chan_1_norm)

    return diff_chan_0, diff_chan_1


def _flow_diff(difference, patch_size, threshold, normalization_value):
    """
    difference between source and test
    :param patch_size:
    :param patch_size: optional, pixel value of flow_b is an average in patch size; odd number size plz
    :param threshold: optional, if thresholding should be applied
    :param normalization_value: used to normalize the difference after smoothing
    :return: result
    """

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

    difference /= normalization_value

    if threshold:
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


def is_anomalous(arguments, anomaly_input, anomaly_thresholds):
    """
    Is flow difference anomalous?

    :param arguments: anomaly_patch_size, pad_disc_output
    :param anomaly_input: np array
    :param anomaly_thresholds: float value
    :return:
    """
    patch_size_row = arguments.anomaly_patch_size[0]
    patch_size_col = arguments.anomaly_patch_size[1]

    if arguments.pad_model_output:
        anomaly_input = np.pad(anomaly_input, pad_width=((0, 0), (1, 1)), mode='constant', constant_values=(0, 0))

    labels = np.full((len(anomaly_thresholds)), fill_value=LABEL_NORMAL, dtype=int)
    row_idx = 0
    for i in range(anomaly_input.shape[0] // patch_size_row):
        col_idx = 0
        for j in range(anomaly_input.shape[1] // patch_size_col):
            z = anomaly_input[row_idx:row_idx + patch_size_row, col_idx:col_idx + patch_size_col]
            avg_diff = np.average(z)

            for k in range(len(anomaly_thresholds)):
                threshold = anomaly_thresholds[k]
                if avg_diff > threshold:
                    labels[k] = LABEL_ANOMALOUS

            # if all are anomalous, short-circuit loop
            if not np.any(labels):
                return labels

            col_idx += patch_size_col

        row_idx += patch_size_row

    return labels

