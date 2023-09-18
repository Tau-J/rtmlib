import numpy as np

from .. import YOLOX, RTMPose


class Body:

    def __init__(
            self,
            det: str = './yolox_l.onnx',
            det_input_size: tuple = (640, 640),
            pose:
        str = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip',  # noqa
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
