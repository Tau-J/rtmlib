# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any

import cv2

OPENCV_DNN_SETTINGS = {
    'cpu': (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),

    # You need to manually build OpenCV through cmake to work with your GPU.
    'cuda': (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
}


class BaseTool(metaclass=ABCMeta):

    def __init__(self,
                 onnx_model: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 device: str = 'cpu'):

        backend, providers = OPENCV_DNN_SETTINGS[device]

        self.session = cv2.dnn.readNetFromONNX(onnx_model)
        self.session.setPreferableBackend(backend)
        self.session.setPreferableTarget(providers)

        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError
