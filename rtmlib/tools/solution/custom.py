'''
Example:

import cv2

from rtmlib import Custom, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture('./demo.mp4')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

# Example: BodyWithFeet in balanced mode
custom = Custom(to_openpose=openpose_skeleton,
                det_class='YOLOX',
                det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip', # noqa
                det_input_size=(640, 640),
                pose_class='RTMPose',
                pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip', # noqa
                pose_input_size=(192, 256),
                backend=backend,
                device=device)

# # Example: RTMO in balanced mode
# custom = Custom(to_openpose=openpose_skeleton,
#                 pose_class='RTMO',
#                 pose='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip', # noqa
#                 pose_input_size=(640,640),
#                 backend=backend,
#                 device=device)
                      
# # Example: Hand in lightweight mode
# custom = Custom(to_openpose=openpose_skeleton,
#                 det_class='RTMDet',
#                 det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip',
#                 det_input_size=(320,320),
#                 pose_class='RTMPose',
#                 pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip',
#                 pose_input_size=(256, 256),
#                 backend=backend,
#                 device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = custom(frame)

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
import importlib
rtmlib_module = importlib.import_module("rtmlib")


class Custom:
    def __init__(self,
                 det_class: str = None,
                 det: str = None,
                 det_input_size: tuple = (640, 640),
                 pose_class: str = None,
                 pose: str = None,
                 pose_input_size: tuple = (192, 256),
                 mode: str = None,
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):

        if det_class is not None:
            try:
                det_class = getattr(rtmlib_module, det_class)
                self.det_model = det_class(det,
                                    model_input_size=det_input_size,
                                    backend=backend,
                                    device=device)
                self.one_stage = False

            except ImportError:
                raise ImportError(f'{det_class} is not supported by rtmlib.')
        else:
            self.one_stage = True

        if pose_class is not None:
            try:
                pose_class = getattr(rtmlib_module, pose_class)
                self.pose_model = pose_class(pose,
                                    model_input_size=pose_input_size,
                                    to_openpose=to_openpose,
                                    backend=backend,
                                    device=device)
            except ImportError:
                raise ImportError(f'{pose_class} is not supported by rtmlib.')

    def __call__(self, image: np.ndarray):
        if self.one_stage:
            keypoints, scores = self.pose_model(image)
        else:
            bboxes = self.det_model(image)
            keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores
