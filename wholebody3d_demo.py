import time

import cv2

from rtmlib import PoseTracker, Wholebody3d, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture(0)

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody3d = PoseTracker(
    Wholebody3d,
    det_frequency=7,
    tracking=False,
    to_openpose=openpose_skeleton,
    backend=backend,
    device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break
    s = time.time()

    keypoints, scores, keypoints_simcc, keypoints_2d = wholebody3d(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints_2d,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.5)

    cv2.imshow('img', img_show)
    cv2.waitKey(10)
