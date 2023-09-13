# rtmlib

rtmlib is a super lightweight library to conduct pose estimation based on [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose) models **WITHOUT** any dependencies like mmcv, mmpose, mmdet, etc.

Basically, rtmlib only requires these dependencies:

- numpy
- opencv-python
- opencv-contrib-python

Optionally, you can use other common backends like pytorch, onnxruntime, tensorrt to accelerate the inference process.

## Installation

```shell
git clone https://github.com/Tau-J/rtmlib.git
cd rtmlib

pip install -r requirements.txt

pip install -e .
```

## TODO

- [x] Support MMPose-style skeleton visualization
- [ ] Support OpenPose-style skeleton visualization
- [ ] Support Pytorch backend
- [ ] Support ONNXRuntime backend
- [ ] Support TensorRT backend

## Model Zoo

### Detection

|  Model  |                                              Download                                              |
| :-----: | :------------------------------------------------------------------------------------------------: |
| YOLOX-l | [Google Drive](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing) |

### Pose Estimation

|   Model   |                                                                     Download                                                                     |
| :-------: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMPose-m | [MMPose](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip) |

## Demo

Here is a simple demo to show how to use rtmlib to conduct pose estimation on a single image.

```python
import cv2

from rtmlib import YOLOX, RTMPose, draw_bbox, draw_skeleton

device = 'cpu'
img = cv2.imread('./demo.jpg')

det_model = YOLOX('./yolox_l.onnx',
                  model_input_size=(640, 640),
                  device=device)
pose_model = RTMPose('./rtmpose.onnx',
                     model_input_size=(192, 256),
                     device=device)

bboxes = det_model(img)
results = pose_model(img, bboxes=bboxes)

# visualize
img_show = draw_bbox(img.copy(), bboxes)

for each in results:
        keypoints = each['keypoints']
        scores = each['scores']

        img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)

cv2.imshow('img', img_show)
cv2.waitKey()
```

Here is also a demo to show how to use rtmlib to conduct pose estimation on a video.

```shell
python demo.py
```

<img width="713" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/c9e6fbaa-00f0-4961-ac87-d881edca778b">

## Acknowledgement

Our code is based on [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose) and [DWPose](https://github.com/IDEA-Research/DWPose/tree/opencv_onnx)
