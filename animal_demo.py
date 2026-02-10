import cv2

from rtmlib import Animal, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture(0)

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
