'''
Example:

import cv2

from rtmlib import Animal, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture('./demo.mp4')

animal = Animal(backend=backend, device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = animal(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=True,
                             kpt_thr=0.43)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)

'''
import numpy as np


class Animal:
    MODE = {
        'performance': {
            'det':
            'https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_l_v142_704x704.onnx',  # noqa
            'det_input_size': (704, 704),
            'pose':
            'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-l-apt36k.onnx',  # noqa
            'pose_input_size': (192, 256),
        },
        'lightweight': {
            'det':
            'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx',  # noqa # seems to not give class ids
            'det_input_size': (416, 416),
            'pose':
            'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-s-apt36k.onnx',  # noqa
            'pose_input_size': (192, 256),
        },
        'balanced': {
            'det':
            'https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_s_v142_512x512.onnx',  # noqa
            'det_input_size': (512, 512),
            'pose':
            'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx',  # noqa
            'pose_input_size': (192, 256),
        }
    }

    def __init__(self,
                 det: str = None,
                 det_categories: list = None,
                 det_input_size: tuple = (640, 640),
                 pose: str = None,
                 pose_input_size: tuple = (192, 256),
                 mode: str = 'balanced',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):

        from .. import YOLOX, RFDETR, ViTPose

        if pose is None:
            pose = self.MODE[mode]['pose']
            pose_input_size = self.MODE[mode]['pose_input_size']

        if det is None:
            det = self.MODE[mode]['det']
            det_input_size = self.MODE[mode]['det_input_size']

        if 'yolox' in det:
            self.det_model = YOLOX(det,
                                det_mode='multiclass',
                                model_input_size=det_input_size,
                                backend=backend,
                                device=device)
        elif 'rfdetr' in det:
            self.det_model = RFDETR(det,
                                det_mode='multiclass',
                                model_input_size=det_input_size,
                                backend=backend,
                                device=device)
        else:
            raise ValueError(
                f"Unsupported detection model: {det}. Please use a YOLOX or RFDETR model."
            )
        self.det_categories = det_categories
        self.pose_model = ViTPose(pose,
                                  model_input_size=pose_input_size,
                                  to_openpose=False,
                                  backend=backend,
                                  device=device)

    def __call__(self, image: np.ndarray):
        if self.det_categories:
            bboxes, classes = self.det_model(image)
            bboxes = [
                bbox for bbox, cls in zip(bboxes, classes)
                if cls in self.det_categories
            ]
        else:
            bboxes, _ = self.det_model(image)
        keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores
