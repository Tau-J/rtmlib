import time

import cv2

from rtmlib import Body, draw_skeleton, draw_skeleton_fixColor

# import numpy as np
import random
device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture(0)
cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

body = Body(
    pose='rtmo',
    to_openpose=openpose_skeleton,
    mode='balanced',  # balanced, performance, lightweight
    backend=backend,
    device=device)

frame_idx = 0

def generate_random_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        # 生成一个随机颜色 (B, G, R)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors

# all_color = generate_random_colors(20)
all_color = [(255, 128, 0), (51, 153, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
skel_height = 270
skel_width = 480
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break
    s = time.time()
    keypoints, scores = body(frame)
    det_time = time.time() - s
    print('det: ', det_time * 1000)
    img_show = frame.copy()
    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)
    img_show = draw_skeleton_fixColor(img_show,
                             all_color, skel_width, skel_height,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.3,
                             line_width=2)
    # dim = (int(frame.shape[1] * 0.5), int((frame.shape[0] + skel_height) * 0.5))
    # img_show = cv2.resize(img_show, dim, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('img', img_show)
    cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
    if cv2.waitKey(5) == 'q':
        break
