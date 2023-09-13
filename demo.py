import time

import cv2

from rtmlib import YOLOX, RTMPose, draw_bbox, draw_skeleton

# import numpy as np

device = 'cpu'

# img = cv2.imread('./demo.jpg')
cap = cv2.VideoCapture('./demo.jpg')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

det_model = YOLOX('./yolox_l.onnx', model_input_size=(640, 640), device=device)
pose_model = RTMPose('./rtmpose.onnx',
                     model_input_size=(192, 256),
                     to_openpose=openpose_skeleton,
                     device=device)

video_writer = None
pred_instances_list = []
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break
    s = time.time()
    bboxes = det_model(frame)
    det_time = time.time() - s
    print('det: ', det_time)

    img_show = frame.copy()
    img_show = draw_bbox(img_show, bboxes)

    s = time.time()
    keypoints, scores = pose_model(frame, bboxes=bboxes)
    pose_time = time.time() - s
    print('pose: ', pose_time)

    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)

    img_show = draw_skeleton(
        img_show,
        keypoints,
        scores,
        skeleton='openpose18' if openpose_skeleton else 'coco17',
        kpt_thr=0.5)

    cv2.imshow('img', img_show)
    cv2.waitKey()
