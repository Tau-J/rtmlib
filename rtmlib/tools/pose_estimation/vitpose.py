from typing import List, Tuple

import numpy as np

from ..base import BaseTool
from .post_processings import convert_coco_to_openpose, post_dark_udp
from .pre_processings import bbox_xyxy2cs, top_down_affine


class ViTPose(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (192, 256),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super().__init__(onnx_model, model_input_size, mean, std, backend,
                         device)
        self.to_openpose = to_openpose

    def __call__(self, image: np.ndarray, bboxes: list = []):
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        keypoints, scores = [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(image, bbox)
            outputs = self.inference(img)
            kpts, score = self.postprocess(outputs, center, scale)

            keypoints.append(kpts)
            scores.append(score)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores

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

    def postprocess(self,
                    outputs: List[np.ndarray],
                    center: Tuple[int, int],
                    scale: Tuple[int, int],
                    score_threshold: float = 0.0,
                    dark_kernel: int = 11) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for ViTPose model output.

        Args:
            outputs (np.ndarray): Output heatmaps of ViTPose model.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            score_threshold (float): Threshold to filter out low score keypoints.
            dark_kernel (int): Kernel size for DARK post-processing.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # extract heatmaps
        heatmaps = outputs[0]
        N, K, H, W = heatmaps.shape

        # get initial keypoints and scores from heatmaps
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        scores = np.max(heatmaps_reshaped, 2).reshape((N, K, 1))

        # convert flattened indices to 2D coordinates
        keypoints = np.tile(idx, (1, 1, 2)).astype(np.float32)
        keypoints[..., 0] = keypoints[..., 0] % W
        keypoints[..., 1] = keypoints[..., 1] // W

        # filter low-confidence keypoints
        keypoints = np.where(
            np.tile(scores, (1, 1, 2)) > score_threshold, keypoints, -1)

        # apply DARK post-processing for sub-pixel accuracy
        keypoints = post_dark_udp(keypoints, heatmaps, kernel=dark_kernel)

        # rescale keypoints
        keypoints = keypoints / (np.array([W, H]) - 1) * scale
        keypoints = keypoints + center - scale / 2

        scores = np.squeeze(scores, axis=2)

        return keypoints, scores
