import numpy as np

from .. import YOLOX, RTMPose


class Wholebody:

    def __init__(self, to_openpose: bool = False, device: str = 'cpu'):
        self.det_model = YOLOX('./yolox_l.onnx', device=device)
        self.pose_model = RTMPose('./dwpose-l-384x288.onnx',
                                  model_input_size=(288, 384),
                                  to_openpose=to_openpose,
                                  device=device)

    def __call__(self, image: np.ndarray):
        bboxes = self.det_model(image)
        keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores
