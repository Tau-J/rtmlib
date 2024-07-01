import time
import cv2
from rtmlib import BodyWithFeet, PoseTracker, draw_skeleton

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture(r"D:\rtmlib_Pose2Sim\demo.jpg")  # Video file path

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

body_feet_tracker = PoseTracker(
    BodyWithFeet,
    det_frequency=7,
    to_openpose=openpose_skeleton,
    mode='performance',  # balanced, performance, lightweight
    backend=backend,
    device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break
    s = time.time()
    keypoints, scores = body_feet_tracker(frame)
    det_time = time.time() - s
    print('det: ', det_time)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.3,
                             line_width=3)

    img_show = cv2.resize(img_show, (960, 640))
    while True:
        cv2.imshow('Body and Feet Pose Estimation', img_show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Press 'q' to exit
            break

cap.release()
cv2.destroyAllWindows()
