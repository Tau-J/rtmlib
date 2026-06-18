import cv2

from rtmlib import Custom, draw_skeleton

MODE = {
    'performance': {
        'det': 'https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_l_v142_704x704.onnx',
        'det_input_size': (704, 704),
    },
    'balanced': {
        'det': 'https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_m_v142_576x576.onnx',
        'det_input_size': (576, 576),
    },
    'lightweight': {
        'det': 'https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_s_v142_512x512.onnx',
        'det_input_size': (512, 512),
    },
}

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
mode = 'balanced'  # performance, balanced, lightweight

cap = cv2.VideoCapture(0)
openpose_skeleton = False

custom = Custom(
    to_openpose=openpose_skeleton,
    det_class='RFDETR',
    det=MODE[mode]['det'],
    det_input_size=MODE[mode]['det_input_size'],
    pose_class='RTMPose',
    pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip',
    pose_input_size=(192, 256),
    backend=backend,
    device=device,
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    keypoints, scores = custom(frame)

    img_show = draw_skeleton(
        frame.copy(),
        keypoints,
        scores,
        openpose_skeleton=openpose_skeleton,
        kpt_thr=0.3,
    )
    img_show = cv2.resize(img_show, (960, 640))

    cv2.imshow('Custom RFDETR Pose Demo', img_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
