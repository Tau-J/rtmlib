# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import cv2
import numpy as np

from ..base import BaseTool


class RTMDet(BaseTool):

    def __init__(self,
                 onnx_model: str = 'rtmdet-m-640x640',
                 model_input_size: tuple = (640, 640),
                 device: str = 'cpu'):
        super().__init__(onnx_model, model_input_size, device)
        raise NotImplementedError

    def __call__(self, image: np.ndarray):
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)
        # results = self.postprocess(outputs, center, scale)

        return outputs

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

        return padded_img, ratio

    def inference(self, img: np.ndarray):
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img, dtype=np.float32)

        input = [img]
        outNames = self.session.getUnconnectedOutLayersNames()
        self.session.setInput(input)
        outputs = self.session.forward(outNames)
        return outputs

    def postprocess(
            self,
            outputs: List[np.ndarray],
            model_input_size: Tuple[int, int],
            center: Tuple[int, int],
            scale: Tuple[int, int],
            simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        pass
