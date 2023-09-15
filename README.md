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

# [optional]
# pip install onnxruntime
# or
# pip install onnxruntime-gpu
```

## TODO

- [x] Support MMPose-style skeleton visualization
- [x] Support OpenPose-style skeleton visualization
- [x] Support WholeBody
- [x] Support ONNXRuntime backend
- [x] Support auto download and cache models
- [ ] Lightweight models
- [ ] Support alias to choose model
- [ ] Support PoseTracker proposed in RTMPose
- [ ] Support TensorRT backend
- [ ] Gradio interface
- [ ] Compatible with Controlnet

## Model Zoo

Here are some models we have converted to onnx format.

|                                              Det                                              |                                                    Pose                                                     |
| :-------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
| [YOLOX-l](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing) | [RTMPose-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip) |

## Demo

Here is a simple demo to show how to use rtmlib to conduct pose estimation on a single image.

```python
import cv2

from rtmlib import YOLOX, RTMPose, draw_bbox, draw_skeleton

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime
img = cv2.imread('./demo.jpg')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

det_model = YOLOX('./yolox_l.onnx',
                  model_input_size=(640, 640),
                  backend=backend,
                  device=device)
pose_model = RTMPose('./rtmpose.onnx',
                     model_input_size=(192, 256),
                     to_openpose=openpose_skeleton,
                     backend=backend,
                     device=device)

bboxes = det_model(img)
keypoints, scores = pose_model(img, bboxes=bboxes)

# visualize
img_show = draw_bbox(img.copy(), bboxes)

# if you want to use black background instead of original image,
# img_show = np.zeros(img_show.shape, dtype=np.uint8)

img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)


cv2.imshow('img', img_show)
cv2.waitKey()
```

Here is also a demo to show how to use rtmlib to conduct wholebody pose estimation.

```shell
import time

import cv2
from rtmlib import Wholebody, draw_skeleton

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture('./demo.jpg')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
                      backend=backend, device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break
    s = time.time()
    keypoints, scores = wholebody(frame)
    det_time = time.time() - s
    print('det: ', det_time)

    img_show = frame.copy()

    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)

    img_show = draw_skeleton(
        img_show,
        keypoints,
        scores,
        openpose_skeleton=openpose_skeleton,
        kpt_thr=0.43)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey()

```

### Visualization

|                                            MMPose-style                                             |                                            OpenPose-style                                             |
| :-------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/c9e6fbaa-00f0-4961-ac87-d881edca778b"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/9afc996a-59e6-4200-a655-59dae10b46c4"> |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/b12e5f60-fec0-42a1-b7b6-365e93894fb1"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/5acf7431-6ef0-44a8-ae52-9d8c8cb988c9"> |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/091b8ce3-32d5-463b-9f41-5c683afa7a11"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/4ffc7be1-50d6-44ff-8c6b-22ea8975aad4"> |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/6fddfc14-7519-42eb-a7a4-98bf5441f324"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/2523e568-e0c3-4c2e-8e54-d1a67100c537"> |

### Citation

```
@misc{rtmlib,
  title={rtmlib},
  author={Tao Jiang},
  year={2023},
}

@misc{https://doi.org/10.48550/arxiv.2303.07399,
  doi = {10.48550/ARXIV.2303.07399},
  url = {https://arxiv.org/abs/2303.07399},
  author = {Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningsheng and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Acknowledgement

Our code is based on these repos:
- [MMPose](https://github.com/open-mmlab/mmpose)
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose)
- [DWPose](https://github.com/IDEA-Research/DWPose/tree/opencv_onnx)
