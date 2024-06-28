import time
import cv2
from rtmlib import Body_and_Feet, PoseTracker, draw_skeleton

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture(0)  # Video file path

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

body_feet_tracker = PoseTracker(
    Body_and_Feet,
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
                             line_width=5)

    img_show = cv2.resize(img_show, (960, 640))
    cv2.imshow('Body and Feet Pose Estimation', img_show)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()