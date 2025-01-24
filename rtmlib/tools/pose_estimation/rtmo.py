from typing import List, Tuple

import cv2
import numpy as np

from ..base import BaseTool
from .post_processings import convert_coco_to_openpose
from ..object_detection.post_processings import multiclass_nms


class RTMO(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (640, 640),
                 mean: tuple = None,
                 std: tuple = None,
                 nms_thr: float = 0.45,
                 score_thr: float = 0.7,
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super().__init__(onnx_model, model_input_size, mean, std, backend,
                         device)
        self.to_openpose = to_openpose
        self.nms_thr = nms_thr
        self.score_thr = score_thr

    def __call__(self, image: np.ndarray, nms_thr: float = None, score_thr: float = None):
        nms_thr = nms_thr if nms_thr is not None else self.nms_thr
        score_thr = score_thr if score_thr is not None else self.score_thr

        image, ratio = self.preprocess(image)
        outputs = self.inference(image)

        keypoints, scores = self.postprocess(outputs, ratio, nms_thr, score_thr)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        if len(img.shape) == 3:
            padded_img = np.ones(
                (self.model_input_size[0], self.model_input_size[1], 3),
                dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(self.model_input_size[0] / img.shape[0],
                    self.model_input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            padded_img = (padded_img - self.mean) / self.std

        return padded_img, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.,
        nms_thr: float = None,
        score_thr: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMO model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMO model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """
        det_outputs, pose_outputs = outputs

        # onnx contains nms module (?)
        final_boxes, final_scores = (det_outputs[0, :, :4], det_outputs[0, :, 4])
        final_boxes /= ratio
        keypoints, scores = pose_outputs[0, :, :, :2], pose_outputs[0, :, :, 2]
        keypoints = keypoints / ratio

        # apply nms
        dets, keep = multiclass_nms(final_boxes, 
                    final_scores[:, np.newaxis],
                    nms_thr=nms_thr,
                    score_thr=score_thr)
        if keep is not None:
            keypoints = keypoints[keep]
            scores = scores[keep]
        else:
            keypoints = np.expand_dims(np.zeros_like(keypoints[0]), axis=0)
            scores = np.expand_dims(np.zeros_like(scores[0]), axis=0)

        return keypoints, scores
