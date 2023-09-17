from typing import List, Optional

import numpy as np

from .. import YOLOX, RTMPose
from .types import BodyResult, Keypoint, PoseResult


class Wholebody:

    def __init__(
            self,
            det: str = './yolox_l.onnx',
            det_input_size: tuple = (640, 640),
            pose:
        str = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip',  # noqa
            pose_input_size: tuple = (288, 384),
            to_openpose: bool = False,
            backend: str = 'opencv',
            device: str = 'cpu'):

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
            part: np.ndarray
        ) -> Optional[List[Optional[Keypoint]]]:
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

            # Openpose face consists of 70 points in total, while DWPose only
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
