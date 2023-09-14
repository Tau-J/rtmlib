import numpy as np

from .. import YOLOX, RTMPose


class Wholebody:

    def __init__(self,
                 to_openpose: bool = False,
                 backend: str = 'opencv',
                 device: str = 'cpu'):
        det = './yolox_l.onnx'
        pose = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip'  # noqa
        self.det_model = YOLOX(det, backend=backend, device=device)
        self.pose_model = RTMPose(pose,
                                  model_input_size=(288, 384),
                                  to_openpose=to_openpose,
                                  backend=backend,
                                  device=device)

    def __call__(self, image: np.ndarray):
        bboxes = self.det_model(image)
        keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores
