'''
import time

import cv2

from rtmlib import PoseTracker, Wholebody3d, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture(0)

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody3d = PoseTracker(
    Wholebody3d,
    det_frequency=7,
    tracking=False,
    to_openpose=openpose_skeleton,
    backend=backend,
    device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break
    s = time.time()

    keypoints, scores, keypoints_simcc, keypoints_2d = wholebody3d(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints_2d,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.5)

    cv2.imshow('img', img_show)
    cv2.waitKey(10)
'''

import numpy as np

from .. import YOLOX, RTMPose3d

class Wholebody3d:

    MODE = {
        'balanced': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx',  # noqa
            'pose_input_size': (288, 384),
        }
    }

    def __init__(self,
                 det: str = None,
                 det_input_size: tuple = (640, 640),
                 pose: str = None,
                 pose_input_size: tuple = (288, 384),
                 mode: str = 'balanced',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):

        if det is None:
            det = self.MODE[mode]['det']
            det_input_size = self.MODE[mode]['det_input_size']

        if pose is None:
            pose = self.MODE[mode]['pose']
            pose_input_size = self.MODE[mode]['pose_input_size']

        self.det_model = YOLOX(det,
                               model_input_size=det_input_size,
                               backend=backend,
                               device=device)
        self.pose_model = RTMPose3d(pose,
                                  model_input_size=pose_input_size,
                                  to_openpose=to_openpose,
                                  backend=backend,
                                  device=device)

    def __call__(self, image: np.ndarray):
        bboxes = self.det_model(image)
        keypoints, scores, keypoints_simcc, keypoints_2d = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores, keypoints_simcc, keypoints_2d
