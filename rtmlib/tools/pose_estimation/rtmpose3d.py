from typing import List, Tuple, Optional

import numpy as np

from ..base import BaseTool
from .post_processings import get_simcc_maximum3d
from .pre_processings import bbox_xyxy2cs, top_down_affine


class RTMPose3d(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (288, 384),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu',
                 z_range: Optional[int] = None):
        super().__init__(onnx_model, model_input_size, mean, std, backend,
                         device)
        self.to_openpose = to_openpose

        self.z_range = z_range if z_range is not None else 2.1744869

    def __call__(self, image: np.ndarray, bboxes: list = []):
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        keypoints, scores, keypoints_simcc, keypoints_2d = [], [], [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(image, bbox)
            outputs = self.inference(img)
            kpts, score, kpts_simcc, kpts_2d = self.postprocess(outputs, center, scale)
            keypoints.append(kpts)
            scores.append(score)
            keypoints_simcc.append(kpts_simcc)
            keypoints_2d.append(kpts_2d)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)
        keypoints_simcc = np.concatenate(keypoints_simcc, axis=0)
        keypoints_2d = np.concatenate(keypoints_2d, axis=0)

        # Not Implemented
        # if self.to_openpose:
        #     keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores, keypoints_simcc, keypoints_2d

    def preprocess(self, img: np.ndarray, bbox: list):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale,
                                             center, img)
        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            resized_img = (resized_img - self.mean) / self.std

        return resized_img, center, scale

    def postprocess(
            self,
            outputs: List[np.ndarray],
            center: Tuple[int, int],
            scale: Tuple[int, int],
            simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled 3D keypoints.
            - scores (np.ndarray): Model predict scores.
            - keypoints_simcc (np.ndarray): Rescaled 3D keypoints in simcc.
            - keypoints_2d (np.ndarray): Rescaled 2D keypoints
        """
        # decode simcc
        simcc_x, simcc_y, simcc_z = outputs
        locs, scores = get_simcc_maximum3d(simcc_x, simcc_y, simcc_z)

        keypoints = locs / simcc_split_ratio
        keypoints_simcc = keypoints.copy()
        keypoints_z = keypoints[..., 2:3]

        keypoints[..., 2:3] = (keypoints_z /
                               (self.model_input_size[-1] / 2) - 1) * self.z_range

        keypoints_2d = keypoints[..., :2].copy()
        keypoints_2d = keypoints_2d / self.model_input_size * scale \
                + center - 0.5 * scale

        return keypoints, scores, keypoints_simcc, keypoints_2d
