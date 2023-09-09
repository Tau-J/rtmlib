import time

import cv2

from rtmlib import YOLOX, RTMPose

device = 'cpu'
# img = cv2.imread('./demo.jpg')
cap = cv2.VideoCapture('./demo.jpg')

det_model = YOLOX('./yolox_l.onnx',
                  model_input_size=(640, 640),
                  device=device)
pose_model = RTMPose('./rtmpose.onnx',
                     model_input_size=(192, 256),
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
    res = det_model(frame)
    det_time = time.time() - s
    print('det: ', det_time)
    img_show = frame.copy()
    for bbox in res:
        img_show = cv2.rectangle(img_show,
                                 (int(bbox[0]), int(bbox[1])),
                                 (int(bbox[2]), int(bbox[3])),
                                 (0, 255, 0),
                                 2)
    s = time.time()
    res = pose_model(frame, bboxes=res)
    pose_time = time.time() - s
    print('pose: ', pose_time)
    for each in res:
        p = each['keypoints']
        K = p.shape[1]
        for i in range(K):
            img_show = cv2.circle(img_show,
                                  (int(p[0, i, 0]), int(p[0, i, 1])),
                                  2,
                                  (0, 0, 255),
                                  2)

    cv2.imshow('img', img_show)
    cv2.waitKey()