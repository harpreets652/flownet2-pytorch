import glob
import os
import numpy as np
import torch.utils.data as data
import torch
import cv2

import utils.flow_utils as fu

# this should be a parameter
SAMPLING_STRIDE = 1


class VideoFiles(data.Dataset):
    def __init__(self, args, root='', file_pattern="*.mov"):
        self.video_files_list = glob.glob(os.path.join(root, file_pattern), recursive=False)

        self.video_files_list.sort()
        self.size = len(self.video_files_list)
        return

    def __getitem__(self, index):
        return np.random.choice(self.video_files_list, 1)[0]

    def __len__(self):
        return self.size


class VideoFileDataJIT(data.Dataset):
    def __init__(self, args, file_path):
        crop_size = tuple(args.crop_size)
        if crop_size[0] < 0 or crop_size[1] < 0:
            crop_size = (512, 384)

        self.video_frames = read_video_frames(file_path, crop_size)
        self.size = self.video_frames.shape[0] - SAMPLING_STRIDE
        return

    def __getitem__(self, index):
        image_a = self.video_frames[index]
        image_b = self.video_frames[index + SAMPLING_STRIDE]

        input_images, flow = generate_flow_frames(image_a, image_b, "none")

        images_tensor = torch.from_numpy(input_images)
        flow_tensor = torch.from_numpy(flow)

        return [images_tensor], [flow_tensor]

    def __len__(self):
        return self.size


class VideoFileData(data.Dataset):
    def __init__(self, args, file_path):
        crop_size = tuple(args.crop_size)
        if crop_size[0] < 0 or crop_size[1] < 0:
            crop_size = (512, 384)

        video_frames = read_video_frames(file_path, crop_size)

        self.images, self.flows = _generate_flow_frames_stride(video_frames, SAMPLING_STRIDE, "none")

        self.size = self.images.shape[0]
        return

    def __getitem__(self, index):
        images = self.images[index]
        flow = self.flows[index]

        images_tensor = torch.from_numpy(images)
        flow_tensor = torch.from_numpy(flow)

        return [images_tensor], [flow_tensor]

    def __len__(self):
        return self.size


def _generate_flow_frames_stride(video_frames, stride, flow_norm_style):
    """
    Generate image-flow pairs for CGAN without masking

    :param video_frames: input video frames
    :param stride: sampling stride
    :param flow_norm_style: "clip|mag|viz|none"
    :return:
    """

    optical_flow_frames = []
    image_frames = []

    i = 0
    while i + stride < video_frames.shape[0]:
        frame_a = video_frames[i, :, :]
        frame_b = video_frames[i + stride, :, :]

        stacked_frames, flow = generate_flow_frames(frame_a, frame_b, flow_norm_style)

        image_frames.append(stacked_frames)
        optical_flow_frames.append(flow)

        i += stride

    return np.array(image_frames), np.array(optical_flow_frames)


def generate_flow_frames(image_a, image_b, flow_norm_style):
    flow_norm_clip = 25.0

    def normalize_scale(input_frame):
        input_frame = input_frame.astype("float32")

        # [0, 1]
        # input_frame /= 255.0

        # [0, 1] => [-1, 1]
        # input_frame = input_frame * 2 - 1

        return input_frame

    normalized_a = normalize_scale(image_a)
    normalized_b = normalize_scale(image_b)

    # stack frames a, b; input: [channels, num_stacked_frames, rows, cols]
    stacked_frames = np.array([normalized_a, normalized_b]).transpose(3, 0, 1, 2)

    frame_a_gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    frame_b_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    if flow_norm_style == "clip":
        flow = fu.compute_optical_flow_farneback(frame_a_gray, frame_b_gray, False)
        flow.clip(-flow_norm_clip, flow_norm_clip)
        flow /= flow_norm_clip
    elif flow_norm_style == "viz":
        flow = fu.compute_optical_flow_farneback(frame_a_gray, frame_b_gray, False)
        flow_viz = fu.flow_to_image_hsv(flow)
        flow = normalize_scale(flow_viz)
    elif flow_norm_style == "mag":
        flow_mag = fu.compute_optical_flow_farneback(frame_a_gray, frame_b_gray, True)
        flow = flow_mag[:, :, :2]
        mag = flow_mag[:, :, 2]
        flow[:, :, 0] = np.divide(flow[:, :, 0], mag, out=np.zeros_like(flow[:, :, 0]), where=mag != 0)
        flow[:, :, 1] = np.divide(flow[:, :, 1], mag, out=np.zeros_like(flow[:, :, 1]), where=mag != 0)
    else:
        flow = fu.compute_optical_flow_farneback(frame_a_gray, frame_b_gray, False)

    flow = flow.transpose(2, 0, 1)

    return stacked_frames, flow


def read_video_frames(video_file, image_size=(512, 384)):
    input_video = cv2.VideoCapture(video_file)

    if not input_video.isOpened():
        raise RuntimeError(f"Unable to open input video {video_file}")

    video_frames = []

    frame_exists, frame_a = input_video.read()

    if not frame_exists:
        raise RuntimeError(f"No frames in video: {video_file}")

    frame_a = pre_process(frame_a, image_size)
    video_frames.append(frame_a)
    frame_counter = 1

    while True:
        frame_exists, frame_a = input_video.read()

        if not frame_exists:
            break

        frame_a = pre_process(frame_a, image_size)
        video_frames.append(frame_a)
        frame_counter += 1

        # if frame_counter % 4 == 0:
        #     break

    input_video.release()

    return np.array(video_frames)


def pre_process(frame, image_size):
    """
    pre-process frame

    :param image_size: tuple, image size
    :param frame: input frame
    :return: processed frame
    """

    # image shape is [height, width, channel] but resize expects (width, height)
    frame = frame[:576, :, :]
    frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_LINEAR)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame
