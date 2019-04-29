import numpy as np
import cv2

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

    flow = cv2.calcOpticalFlowFarneback(frame_a, frame_b, None, 0.5, 2, 13, 3, 5, 1.1, 0)

    if include_magnitude:
        mag, _ = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        mag = np.expand_dims(mag, -1)
        flow = np.dstack((flow, mag))

    return flow


def show_flow(flow, win_name='', wait_time=0):
    """Show optical flow

    Args:
        flow(str): optical flow to be shown
        win_name(str): the window name
        wait_time(int): value of waitKey param
    """
    flow = read_flow(flow)
    flow_img = flow_2_rgb(flow)
    # flow_img = flow_to_image_hsv(flow)
    cv2.imshow(win_name, flow_img)
    cv2.waitKey(wait_time)
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

    [h, w] = dx.shape

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
