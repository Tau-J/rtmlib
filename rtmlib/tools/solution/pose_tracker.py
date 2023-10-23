'''
Example:

import cv2
from rtmlib import PoseTracker, Wholebody, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

openpose_skeleton = True  # True for openpose-style, False for mmpose-style

cap = cv2.VideoCapture('./demo.mp4')

wholebody = PoseTracker(Wholebody,
                        det_frequency=10,  # detect every 10 frames
                        to_openpose=openpose_skeleton,
                        backend=backend, device=device)

                        frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = wholebody(frame)

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


def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    """Get bounding box from keypoints.

    Args:
        keypoints (np.ndarray): Keypoints of person.
        expansion (float): Expansion ratio of bounding box.

    Returns:
        np.ndarray: Bounding box of person.
    """
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    bbox = np.concatenate([
        center - (center - bbox[:2]) * expansion,
        center + (bbox[2:] - center) * expansion
    ])
    return bbox


class PoseTracker:
    """Pose tracker for wholebody pose estimation.

    Args:
        solution (type): rtmlib solutions, e.g. Wholebody, Body, etc.
        det_frequency (int): Frequency of object detection.
        mode (str): 'performance', 'lightweight', or 'balanced'.
        to_openpose (bool): Whether to use openpose-style skeleton.
        backend (str): Backend of pose estimation model.
        device (str): Device of pose estimation model.
    """

    def __init__(self,
                 solution: type,
                 det_frequency: int = 1,
                 mode: str = 'balanced',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        print('pose tracker', backend, device)
        model = solution(mode=mode,
                         to_openpose=to_openpose,
                         backend=backend,
                         device=device)

        self.det_model = model.det_model
        self.pose_model = model.pose_model

        self.det_frequency = det_frequency
        self.reset()

    def reset(self):
        """Reset pose tracker."""
        self.cnt = 0
        self.instance_list = []

    def __call__(self, image: np.ndarray):

        if self.cnt % self.det_frequency == 0:
            bboxes = self.det_model(image)
        else:
            bboxes = self.instance_list

        keypoints, scores = self.pose_model(image, bboxes=bboxes)

        instances = []
        for kpts in keypoints:
            instances.append(pose_to_bbox(kpts))

        self.instance_list = instances
        self.cnt += 1

        return keypoints, scores
