from abc import ABCMeta, abstractmethod
from typing import Any

import cv2
import numpy as np

RTMLIB_SETTINGS = {
    'opencv': {
        'cpu': (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),

        # You need to manually build OpenCV through cmake
        'cuda': (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
    },
    'onnxruntime': {
        'cpu': 'CPUExecutionProvider',
        'cuda': 'CUDAExecutionProvider'
    },
}


class BaseTool(metaclass=ABCMeta):

    def __init__(self,
                 onnx_model: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 backend: str = 'opencv',
                 device: str = 'cpu'):

        providers = RTMLIB_SETTINGS[backend][device]

        if backend == 'opencv':
            session = cv2.dnn.readNetFromONNX(onnx_model)
            session.setPreferableBackend(providers[0])
            session.setPreferableTarget(providers[1])

        elif backend == 'onnxruntime':
            import onnxruntime as ort
            session = ort.InferenceSession(path_or_bytes=onnx_model,
                                           providers=[providers])

        else:
            raise NotImplementedError

        print(f'load {onnx_model} with {backend} backend')

        self.session = session
        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = backend
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def inference(self, img: np.ndarray):
        """Inference model.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        # build input
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img[None, :, :, :]

        # run model
        if self.backend == 'opencv':
            outNames = self.session.getUnconnectedOutLayersNames()
            self.session.setInput(input)
            outputs = self.session.forward(outNames)
        elif self.backend == 'onnxruntime':
            sess_input = {self.session.get_inputs()[0].name: input}
            sess_output = []
            for out in self.session.get_outputs():
                sess_output.append(out.name)

            outputs = self.session.run(sess_output, sess_input)

        return outputs
