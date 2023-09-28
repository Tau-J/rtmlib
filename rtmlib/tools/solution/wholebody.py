from typing import List, Optional

import numpy as np

from .. import YOLOX, RTMPose
from .types import BodyResult, Keypoint, PoseResult


class Wholebody:

    MODE = {
        'performance': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip',  # noqa
            'pose_input_size': (288, 384),
        },
        'lightweight': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_s_8xb8-300e_humanart-3ef259a7.zip',  # noqa
            'det_input_size': (416, 416),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-256x192-fbef0d61_20230925.zip',  # noqa
            'pose_input_size': (192, 256),
        },
        'balanced': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-256x192-fbef0d61_20230925.zip',  # noqa
            'pose_input_size': (192, 256),
        }
    }

    def __init__(self,
                 det: str = None,
                 det_input_size: tuple = (640, 640),
                 pose: str = None,
                 pose_input_size: tuple = (288, 384),
                 mode: str = 'performance',
                 to_openpose: bool = False,
                 backend: str = 'opencv',
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
        self.pose_model = RTMPose(pose,
                                  model_input_size=pose_input_size,
                                  to_openpose=to_openpose,
                                  backend=backend,
                                  device=device)

    def __call__(self, image: np.ndarray):
        bboxes = self.det_model(image)
        keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores

    @staticmethod
    def format_result(keypoints_info: np.ndarray) -> List[PoseResult]:

        def format_keypoint_part(
                part: np.ndarray) -> Optional[List[Optional[Keypoint]]]:
            keypoints = [
                Keypoint(x, y, score, i) if score >= 0.3 else None
                for i, (x, y, score) in enumerate(part)
            ]
            return (None if all(keypoint is None
                                for keypoint in keypoints) else keypoints)

        def total_score(
                keypoints: Optional[List[Optional[Keypoint]]]) -> float:
            return (sum(
                keypoint.score for keypoint in keypoints
                if keypoint is not None) if keypoints is not None else 0.0)

        pose_results = []

        for instance in keypoints_info:
            body_keypoints = format_keypoint_part(
                instance[:18]) or ([None] * 18)
            left_hand = format_keypoint_part(instance[92:113])
            right_hand = format_keypoint_part(instance[113:134])
            face = format_keypoint_part(instance[24:92])

            # Openpose face consists of 70 points in total, while RTMPose only
            # provides 68 points. Padding the last 2 points.
            if face is not None:
                # left eye
                face.append(body_keypoints[14])
                # right eye
                face.append(body_keypoints[15])

            body = BodyResult(body_keypoints, total_score(body_keypoints),
                              len(body_keypoints))
            pose_results.append(PoseResult(body, left_hand, right_hand, face))

        return pose_results
