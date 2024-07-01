'''
Example:

import cv2
from rtmlib import Halpe26, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture('./demo.mp4')

to_openpose = True  # True for openpose-style, False for mmpose-style

halpe26 = Halpe26(to_openpose=to_openpose,
                  backend=backend,
                  device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = halpe26(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.43)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)

'''

import numpy as np

class BodyWithFeet:
    """
    Halpe26 class for human pose estimation using the Halpe26 keypoint format.
    This class supports different modes of operation and can output in OpenPose format.
    """

    MODE = {
        'performance': {
            'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
            'det_input_size': (640, 640),
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip',
            'pose_input_size': (288, 384),
        },
        'lightweight': {
            'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',
            'det_input_size': (416, 416),
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip',
            'pose_input_size': (192, 256),
        },
        'balanced': {
            'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
            'det_input_size': (640, 640),
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip',
            'pose_input_size': (192, 256),
        }
    }

    def __init__(self,
                 det: str = None,
                 det_input_size: tuple = (640, 640),
                 pose: str = None,
                 pose_input_size: tuple = (192, 256),
                 mode: str = 'balanced',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        """
        Initialize the Halpe26 pose estimation model.

        Args:
            det (str, optional): Path to detection model. If None, uses default based on mode.
            det_input_size (tuple, optional): Input size for detection model. Default is (640, 640).
            pose (str, optional): Path to pose estimation model. If None, uses default based on mode.
            pose_input_size (tuple, optional): Input size for pose model. Default is (192, 256).
            mode (str, optional): Operation mode ('performance', 'lightweight', or 'balanced'). Default is 'balanced'.
            to_openpose (bool, optional): Whether to convert output to OpenPose format. Default is False.
            backend (str, optional): Backend for inference ('onnxruntime' or 'opencv'). Default is 'onnxruntime'.
            device (str, optional): Device for inference ('cpu' or 'cuda'). Default is 'cpu'.
        """
        from .. import YOLOX, RTMPose

        if pose is None:
            pose = self.MODE[mode]['pose']
            pose_input_size = self.MODE[mode]['pose_input_size']

        if det is None:
            det = self.MODE[mode]['det']
            det_input_size = self.MODE[mode]['det_input_size']

        self.det_model = YOLOX(det,
                               model_input_size=det_input_size,
                               backend=backend,
                               device=device)
        self.pose_model = RTMPose(pose,
                                  model_input_size=pose_input_size,
                                  to_openpose=to_openpose,
                                  backend=backend,
                                  device=device)

    def __call__(self, image: np.ndarray):
        """
        Perform pose estimation on the input image.

        Args:
            image (np.ndarray): Input image for pose estimation.

        Returns:
            tuple: A tuple containing:
                - keypoints (np.ndarray): Estimated keypoint coordinates.
                - scores (np.ndarray): Confidence scores for each keypoint.
        """
        bboxes = self.det_model(image)
        keypoints, scores = self.pose_model(image, bboxes=bboxes)
        return keypoints, scores
