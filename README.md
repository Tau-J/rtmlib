# rtmlib

![demo](https://github.com/Tau-J/rtmlib/assets/13503330/b7e8ce8b-3134-43cf-bba6-d81656897289)

`rtmlib` is a super lightweight library to conduct human and animal pose estimation based on [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose) and [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) models **WITHOUT** any dependencies like mmcv, mmpose, mmdet, etc.

Basically, rtmlib only requires these dependencies:

- numpy
- opencv-python
- opencv-contrib-python
- onnxruntime

Optionally, you can use other common backends like opencv, onnxruntime, openvino, tensorrt to accelerate the inference process.

- For openvino users, please add the path `<your python path>\envs\<your env name>\Lib\site-packages\openvino\libs` into your environment path.

## Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Usage](#usage)
   1. [WebUI](#webui)
   2. [APIs](#apis)
4. [Model Zoo](#model-zoo)
   1. [Detectors](#detectors)
   2. [Pose Estimators](#pose-estimators)
   3. [Visualization](#visualization)
5. [Additional Resources](#additional-resources)
   1. [Citation](#citation)
   2. [Acknowledgement](#acknowledgement)

## Installation

- install from pypi:

```shell
pip install rtmlib -i https://pypi.org/simple
```

- install from source code:

```shell
git clone https://github.com/Tau-J/rtmlib.git
cd rtmlib

pip install -r requirements.txt

pip install -e .

# [optional]
# pip install onnxruntime-gpu
# pip install openvino

```

## Quick Start

Here is a simple demo to show how to use rtmlib to conduct pose estimation on a single image.

```python
import cv2
from rtmlib import Wholebody, draw_skeleton

img = cv2.imread('./demo.jpg')

device = 'cpu'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
openpose_skeleton = False  # True for openpose-style (required for animals), False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)
keypoints, scores = wholebody(img)

# visualize
# if you want to use black background instead of original image,
# img = np.zeros(img.shape, dtype=np.uint8)
img = draw_skeleton(img, keypoints, scores, kpt_thr=0.5, to_openpose=openpose_skeleton)
cv2.imshow('img', img)
cv2.waitKey(0)
```

Or on a webcam stream or a video file.

```Python
from rtmlib import Body, Custom, PoseTracker, draw_skeleton
import cv2

cap = cv2.VideoCapture(0)  # for video file instead of webcam, use cap = cv2.VideoCapture('./demo.mp4')

device = 'cpu'
backend = 'onnxruntime'
openpose_skeleton = False

pose_tracker = PoseTracker(Body,
                        mode='balanced',
                        det_frequency=10,  # detect every 10 frames
                        backend=backend, device=device,
                        to_openpose=False)

# # Or with a custom class
# from functools import partial
# custom = partial(Custom,
#                 det_class='YOLOX',
#                 det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
#                 det_input_size=(640, 640),
#                 pose_class='RTMPose',
#                 pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip',
#                 pose_input_size=(192, 256))
# pose_tracker = PoseTracker(custom,
#                         det_frequency=10,
#                         backend=backend, device=device,
#                         to_openpose=False)

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1
    if not success:
        break

    keypoints, scores = pose_tracker(frame)

    img_show = frame.copy()
    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.43)
    cv2.imshow('img', img_show)
    cv2.waitKey(10)
```

## Usage

### WebUI

Run `webui.py`:

```shell
# Please make sure you have installed gradio
# pip install gradio

python webui.py
```

![image](https://github.com/Tau-J/rtmlib/assets/13503330/49ef11a1-a1b5-4a20-a2e1-d49f8be6a25d)

### APIs

- Solutions (High-level APIs)
  - [PoseTracker](/rtmlib/tools/solution/pose_tracker.py)
  - [Wholebody](/rtmlib/tools/solution/wholebody.py)
  - [Body](/rtmlib/tools/solution/body.py)
  - [Body_with_feet](/rtmlib/tools/solution/body_with_feet.py)
  - [Hand](/rtmlib/tools/solution/hand.py)
  - [Animal](/rtmlib/tools/solution/animal.py) (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)
  - [Custom](/rtmlib/tools/solution/custom.py)
  - [Wholebody3d](/rtmlib/tools/solution/wholebody3d.py)
- Detectors (Low-level APIs)
  - [YOLOX](/rtmlib/tools/object_detection/yolox.py) (human and multiclass)
  - [RTMDet](/rtmlib/tools/object_detection/rtmdet.py)
- Pose Estimators (Low-level APIs)
  - [RTMPose](/rtmlib/tools/pose_estimation/rtmpose.py)
    - RTMPose for 17 keypoints
    - RTMO for 17 keypoints (**one-stage**)
    - RTMPose for 21 keypoints (**hand**)
    - RTMPose for 26 keypoints
    - RTMW for 133 keypoints
    - DWPose for 133 keypoints
    - RTMW3D for 133 keypoints (**3D**)
  - [ViTPose](/rtmlib/tools/pose_estimation/vitpose.py)
    - ViTPose for 17 keypoints
    - ViTPose for 17 keypoints (**animal**)
    - ViTPose for 25 keypoints
    - ViTPose for 133 keypoints
- Visualization
  - [draw_bbox](https://github.com/Tau-J/rtmlib/blob/adc69a850f59ba962d81a88cffd3f48cfc5fd1ae/rtmlib/draw.py#L9)
  - [draw_skeleton](https://github.com/Tau-J/rtmlib/blob/adc69a850f59ba962d81a88cffd3f48cfc5fd1ae/rtmlib/draw.py#L16)

For high-level APIs (`Solution`), you can choose to pass `mode` or `det`+`pose` arguments to specify the detector and pose estimator you want to use.

```Python
# By mode
from rtmlib import Wholebody
wholebody = Wholebody(mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend,
                      device=device)

# By det and pose
from rtmlib import Body
body = Body(det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
            det_input_size=(640, 640),
            pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
            pose_input_size=(288, 384),
            backend=backend,
            device=device)

# By det and pose with custom classes
from rtmlib import Custom
# Human pose estimation using YOLOX and RTMPose
custom = Custom(det_class='YOLOX',
               det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.ip',
               det_input_size=(640, 640),
               pose_class='RTMPose',
               pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
               pose_input_size=(192, 256),
               backend=backend,
               device=device)

# Human and animal pose estimation using YOLOX in multiclass mode and ViTPose
# Requires openpose_skeleton = True in draw_skeleton for visualization
custom = Custom(det_class='YOLOX',
               det_mode='multiclass', # or det_categories=[0,23] (for example)
               det='https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx',
               det_input_size=(640, 640),
               pose_class='ViTPose',
               pose='https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx',
               pose_input_size=(192, 256),
               backend=backend,
               device=device)
```

For low-level APIs (`Model`), you can specify the model you want to use by passing the `onnx_model` argument.

```Python
# By onnx_model (.onnx or .zip) by download link or local path
# YOLOX human detector
det_model = YOLOX(onnx_model='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_s_8xb8-300e_humanart-3ef259a7.zip',
                     backend=backend, device=device)

# YOLOX multiclass detector
det_model = YOLOX('https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx',
                     det_mode='multiclass', # or det_categories=[0,1,etc] if you want specific COCO_CLASSES IDs
                     backend=backend, device=device)

# RTMPose pose estimator
pose_model = RTMPose(onnx_model='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
                     backend=backend, device=device)

# ViTPose pose estimator
pose_model = ViTPose(onnx_model='https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx',
                     backend=backend, device=device)
```

## Model Zoo

By defaults, rtmlib will automatically download and apply models with the best performance.

More models can be found in [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose) and [ViTPose](https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx) Model Zoos for pose estimation, and [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime) for multiclass detection.

### Detectors

<details open>
<summary><b>Person</b></summary>

Notes:

- Models trained on HumanArt can detect both real human and cartoon characters.
- Models trained on COCO can only detect real human.

|                                                          ONNX Model                                                           | Input Size | AP (person) |       Description        |
| :---------------------------------------------------------------------------------------------------------------------------: | :--------: | :---------: | :----------------------: |
|                 [YOLOX-l](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing)                 |  640x640   |      -      |     trained on COCO      |
| [YOLOX-nano](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip) |  416x416   |    38.9     | trained on HumanArt+COCO |
| [YOLOX-tiny](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip) |  416x416   |    47.7     | trained on HumanArt+COCO |
|    [YOLOX-s](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_s_8xb8-300e_humanart-3ef259a7.zip)    |  640x640   |    54.6     | trained on HumanArt+COCO |
|    [YOLOX-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip)    |  640x640   |    59.1     | trained on HumanArt+COCO |
|    [YOLOX-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_l_8xb8-300e_humanart-ce1d7a62.zip)    |  640x640   |    60.2     | trained on HumanArt+COCO |
|    [YOLOX-x](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip)    |  640x640   |    61.3     | trained on HumanArt+COCO |

</details>

<details open>
<summary><b>Hand</b></summary>

|                                                          ONNX Model                                                          | Input Size | AP (hand) |      Description      |
| :--------------------------------------------------------------------------------------------------------------------------: | :--------: | :-------: | :-------------------: |
| [RTMDet-nano](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip) |  320x320   |   76.0    | trained on 5 datasets |

<details open>
<summary><b>Multi-class</b></summary>

```
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

|                                                   ONNX Model                                                   | Input Size | AP (COCO classes) |   Description   |
| :------------------------------------------------------------------------------------------------------------: | :--------: | :---------------: | :-------------: |
|     [YOLOX-nano](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx)     |  416x416   |       25.8        | trained on coco |
|      [YOLOX-t](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx)       |  416x416   |       32.8        | trained on coco |
|        [YOLOX-s](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx)        |  640x640   |       40.5        | trained on coco |
|        [YOLOX-m](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx)        |  640x640   |       47.2        | trained on coco |
|        [YOLOX-l](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.onnx)        |  640x640   |       50.1        | trained on coco |
| [YOLOX-Darknet53](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.onnx) |  640x640   |       48.0        | trained on coco |
|        [YOLOX-X](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.onnx)        |  640x640   |       51.5        | trained on coco |

### Pose Estimators

<details open>
<summary><b>Body 17 Keypoints</b></summary>

|                                                                     ONNX Model                                                                      | Input Size | AP (COCO) |      Description      |
| :-------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :-------: | :-------------------: |
| [RTMPose-t](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip) |  256x192   |   65.9    | trained on 7 datasets |
| [RTMPose-s](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip) |  256x192   |   69.7    | trained on 7 datasets |
| [RTMPose-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip) |  256x192   |   74.9    | trained on 7 datasets |
| [RTMPose-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip) |  256x192   |   76.7    | trained on 7 datasets |
| [RTMPose-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip) |  384x288   |   78.3    | trained on 7 datasets |
| [RTMPose-x](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip) |  384x288   |   78.8    | trained on 7 datasets |
|           [RTMO-s](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip)           |  640x640   |   68.6    | trained on 7 datasets |
|          [RTMO-m](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip)           |  640x640   |   72.6    | trained on 7 datasets |
|          [RTMO-l](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip)           |  640x640   |   74.8    | trained on 7 datasets |
|                       [ViTPose++-s](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-s-coco.onnx)                       |  256x192   |   75.8    | trained on 6 datasets |
|                       [ViTPose++-b](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-b-coco.onnx)                       |  256x192   |   77.0    | trained on 6 datasets |
|                       [ViTPose++-l](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-l-coco.onnx)                       |  256x192   |   78.6    | trained on 6 datasets |

</details>

<details open>
<summary><b>Body 25 Keypoints</b></summary>

|                                                 ONNX Model                                                  | Input Size | AP (COCO) |       Description       |
| :---------------------------------------------------------------------------------------------------------: | :--------: | :-------: | :---------------------: |
| [ViTPose-s](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco_25/vitpose-s-coco_25.onnx) |  256x192   |     -     | fine-tuned on COCO+feet |
| [ViTPose-b](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco_25/vitpose-b-coco_25.onnx) |  256x192   |     -     | fine-tuned on COCO+feet |
| [ViTPose-l](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco_25/vitpose-l-coco_25.onnx) |  256x192   |     -     | fine-tuned on COCO+feet |

</details>

<details open>
<summary><b>Body 26 Keypoints</b></summary>

|                                                                          ONNX Model                                                                           | Input Size | AP (Body8) |      Description      |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :--------: | :-------------------: |
|  [RTMPose-t](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.zip)  |  256x192   |    68.0    | trained on 7 datasets |
|  [RTMPose-s](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip)  |  256x192   |    72.0    | trained on 7 datasets |
|  [RTMPose-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip)  |  256x192   |    76.7    | trained on 7 datasets |
|  [RTMPose-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip)  |  256x192   |    78.4    | trained on 7 datasets |
| [RTMPose-x\*](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip) |  384x288   |    80.0    | trained on 7 datasets |

</details>

<details open>
<summary><b>WholeBody 133 Keypoints</b></summary>

|                                                                     ONNX Model                                                                     | Input Size | AP (Whole) |           Description           |
| :------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :--------: | :-----------------------------: |
|                 [ViTPose++-s](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-s-wholebody.onnx)                  |  256x192   |    54.4    |      trained on 6 datasets      |
|                 [ViTPose++-b](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-b-wholebody.onnx)                  |  256x192   |    57.4    |      trained on 6 datasets      |
|                 [ViTPose++-l](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx)                  |  256x192   |    60.6    |      trained on 6 datasets      |
| [DWPose-t](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip) |  256x192   |    48.5    | trained on COCO-Wholebody+UBody |
| [DWPose-s](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.zip) |  256x192   |    53.8    | trained on COCO-Wholebody+UBody |
| [DWPose-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip) |  256x192   |    60.6    | trained on COCO-Wholebody+UBody |
| [DWPose-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.zip) |  256x192   |    63.1    | trained on COCO-Wholebody+UBody |
| [DWPose-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip) |  384x288   |    66.5    | trained on COCO-Wholebody+UBody |
|          [RTMW-m](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip)          |  256x192   |    58.2    |     trained on 14 datasets      |
|          [RTMW-l](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip)          |  256x192   |    66.0    |     trained on 14 datasets      |
|         [RTMW-l\*](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip)         |  384x288   |    70.1    |     trained on 14 datasets      |
|  [RTMW-x\*](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip)   |  384x288   |    70.2    |     trained on 14 datasets      |

</details>

<details open>
<summary><b>WholeBody 133 Keypoints</b></summary>

|                                                          ONNX Model                                                           | Input Size | AP (Whole) |        Description        |
| :---------------------------------------------------------------------------------------------------------------------------: | :--------: | :--------: | :-----------------------: |
| [RTMW3D-x](https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx) |  384x288   |    68.0    | trained on COCO-Wholebody |

</details>

<details open>
<summary><b>Hand</b></summary>

|                                                                       ONNX Model                                                                        | Input Size | **AUC** (Hand56) |      Description      |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :--------------: | :-------------------: |
| [RTMPose-m\*](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip) |  256x256   |       83.9       | trained on 5 datasets |

</details>

<details open>
<summary><b>Face</b></summary>

|                                                                      ONNX Model                                                                      | Input Size | AP (Face6) |      Description      |
| :--------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :--------: | :-------------------: |
| [RTMPose-t\*](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-face6_pt-in1k_120e-256x256-df79d9a5_20230529.zip) |  256x256   |     -      | trained on 6 datasets |
| [RTMPose-s\*](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-face6_pt-in1k_120e-256x256-d779fdef_20230529.zip) |  256x256   |     -      | trained on 6 datasets |
| [RTMPose-m\*](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.zip) |  256x256   |     -      | trained on 6 datasets |

</details>

<details open>
<summary><b>Animal</b></summary>

```
CATEGORIES = ['gorilla', 'spider-monkey', 'howling-monkey', 'zebra', 'elephant', 'hippo', 'raccon', 'rhino', 'giraffe', 'tiger', 'deer', 'lion', 'panda', 'cheetah', 'black-bear', 'polar-bear', 'antelope', 'fox', 'buffalo', 'cow', 'wolf', 'dog', 'sheep', 'cat', 'horse', 'rabbit', 'pig', 'chimpanzee', 'monkey', 'orangutan']
```

|                                                                       ONNX Model                                                                       | Input Size | AP (AP10K) |      Description      |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :--------: | :-------------------: |
| [RTMPose-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip) |  256x256   |    72.2    |   trained on AP-10K   |
|                      [ViTPose++-s](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-s-apt36k.onnx)                       |  256x192   |    74.2    | trained on 6 datasets |
|                      [ViTPose++-b](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx)                       |  256x192   |    75.9    | trained on 6 datasets |
|                      [ViTPose++-l](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-h-apt36k.onnx)                       |  256x192   |    80.8    | trained on 6 datasets |

</details>

### Visualization

|                                            MMPose-style                                             |                                            OpenPose-style                                             |
| :-------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/c9e6fbaa-00f0-4961-ac87-d881edca778b"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/9afc996a-59e6-4200-a655-59dae10b46c4"> |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/b12e5f60-fec0-42a1-b7b6-365e93894fb1"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/5acf7431-6ef0-44a8-ae52-9d8c8cb988c9"> |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/091b8ce3-32d5-463b-9f41-5c683afa7a11"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/4ffc7be1-50d6-44ff-8c6b-22ea8975aad4"> |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/6fddfc14-7519-42eb-a7a4-98bf5441f324"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/2523e568-e0c3-4c2e-8e54-d1a67100c537"> |

## Additional resources

### Citation

```
@misc{rtmlib,
  title={rtmlib},
  author={Jiang, Tao},
  year={2023},
  howpublished = {\url{https://github.com/Tau-J/rtmlib}},
}

@misc{jiang2023,
  doi = {10.48550/ARXIV.2303.07399},
  url = {https://arxiv.org/abs/2303.07399},
  author = {Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningsheng and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}

@misc{lu2023rtmo,
      title={{RTMO}: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
      author={Peng Lu and Tao Jiang and Yining Li and Xiangtai Li and Kai Chen and Wenming Yang},
      year={2023},
      eprint={2312.07526},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{jiang2024rtmwrealtimemultiperson2d,
      title={RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation},
      author={Tao Jiang and Xinchen Xie and Yining Li},
      year={2024},
      eprint={2407.08634},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.08634},
}

@article{xu2022vitpose,
      title={Vitpose: Simple vision transformer baselines for human pose estimation},
      author={Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
      journal={Advances in neural information processing systems},
      volume={35},
      pages={38571--38584},
      year={2022}
}
```

### Acknowledgement

Our code is based on these repos:

- [MMPose](https://github.com/open-mmlab/mmpose)
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose)
- [DWPose](https://github.com/IDEA-Research/DWPose/tree/opencv_onnx)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
