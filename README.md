# rtmlib

![demo](https://github.com/Tau-J/rtmlib/assets/13503330/b7e8ce8b-3134-43cf-bba6-d81656897289)

rtmlib is a super lightweight library to conduct pose estimation based on [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose) models **WITHOUT** any dependencies like mmcv, mmpose, mmdet, etc.

Basically, rtmlib only requires these dependencies:

- numpy
- opencv-python
- opencv-contrib-python
- onnxruntime

Optionally, you can use other common backends like opencv, onnxruntime, openvino, tensorrt to accelerate the inference process.

- For openvino users, please add the path `<your python path>\envs\<your env name>\Lib\site-packages\openvino\libs` into your environment path.

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

device = 'cpu'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
img = cv2.imread('./demo.jpg')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)

keypoints, scores = wholebody(img)

# visualize

# if you want to use black background instead of original image,
# img_show = np.zeros(img_show.shape, dtype=np.uint8)

img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)


cv2.imshow('img', img_show)
cv2.waitKey()
```

## WebUI

Run `webui.py`:

```shell
# Please make sure you have installed gradio
# pip install gradio

python webui.py
```

![image](https://github.com/Tau-J/rtmlib/assets/13503330/49ef11a1-a1b5-4a20-a2e1-d49f8be6a25d)

## APIs

- Solutions (High-level APIs)
  - [Wholebody](/rtmlib/tools/solution/wholebody.py)
  - [Body](/rtmlib/tools/solution/body.py)
  - [Body_with_feet](/rtmlib/tools/solution/body_with_feet.py)
  - [Hand](/rtmlib/tools/solution/hand.py)
  - [PoseTracker](/rtmlib/tools/solution/pose_tracker.py)
- Models (Low-level APIs)
  - [YOLOX](/rtmlib/tools/object_detection/yolox.py)
  - [RTMDet](/rtmlib/tools/object_detection/rtmdet.py)
  - [RTMPose](/rtmlib/tools/pose_estimation/rtmpose.py)
    - RTMPose for 17 keypoints
    - RTMPose for 26 keypoints
    - RTMW for 133 keypoints
    - DWPose for 133 keypoints
    - RTMO for one-stage pose estimation (17 keypoints)
- Visualization
  - [draw_bbox](https://github.com/Tau-J/rtmlib/blob/adc69a850f59ba962d81a88cffd3f48cfc5fd1ae/rtmlib/draw.py#L9)
  - [draw_skeleton](https://github.com/Tau-J/rtmlib/blob/adc69a850f59ba962d81a88cffd3f48cfc5fd1ae/rtmlib/draw.py#L16)

For high-level APIs (`Solution`), you can choose to pass `mode` or `det`+`pose` arguments to specify the detector and pose estimator you want to use.

```Python
# By mode
wholebody = Wholebody(mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend,
                      device=device)

# By det and pose
body = Body(det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
            det_input_size=(640, 640),
            pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
            pose_input_size=(288, 384),
            backend=backend,
            device=device)
```

For low-level APIs (`Model`), you can specify the model you want to use by passing the `onnx_model` argument.

```Python
# By onnx_model (.onnx)
pose_model = RTMPose(onnx_model='/path/to/your_model.onnx',  # download link or local path
                     backend=backend, device=device)

# By onnx_model (.zip)
pose_model = RTMPose(onnx_model='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',  # download link or local path
                     backend=backend, device=device)
```

## Model Zoo

By defaults, rtmlib will automatically download and apply models with the best performance.

More models can be found in [RTMPose Model Zoo](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose).

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

</details>

<details open>
<summary><b>Body 26 Keypoints</b></summary>

|                                                                     ONNX Model                                                                      | Input Size | AUC (Body8) |      Description      |
| :-------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :-------: | :-------------------: |
| [RTMPose-t](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.zip) |  256x192   |   66.35    | trained on 7 datasets |
| [RTMPose-s](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip) |  256x192   |   68.62    | trained on 7 datasets |
| [RTMPose-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip) |  256x192   |   71.91    | trained on 7 datasets |
| [RTMPose-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip) |  256x192   |   73.19    | trained on 7 datasets |
| [RTMPose-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.zip) |  384x288   |   73.56    | trained on 7 datasets |
| [RTMPose-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.zip) |  384x288   |   74.38    | trained on 7 datasets |
| [RTMPose-x](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip) |  384x288   |   74.82    | trained on 7 datasets |

</details>

<details open>
<summary><b>WholeBody 133 Keypoints</b></summary>

|                                                                     ONNX Model                                                                     | Input Size |   AP (Whole)   |           Description           |
| :------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :--: | :-----------------------------: |
| [DWPose-t](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip) |  256x192   | 48.5 | trained on COCO-Wholebody+UBody |
| [DWPose-s](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.zip) |  256x192   | 53.8 | trained on COCO-Wholebody+UBody |
| [DWPose-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip) |  256x192   | 60.6 | trained on COCO-Wholebody+UBody |
| [DWPose-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.zip) |  256x192   | 63.1 | trained on COCO-Wholebody+UBody |
| [DWPose-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip) |  384x288   | 66.5 | trained on COCO-Wholebody+UBody |
|          [RTMW-m](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip)          |  256x192   | 58.2 |     trained on 14 datasets      |
|          [RTMW-l](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip)          |  256x192   | 66.0 |     trained on 14 datasets      |
|          [RTMW-l](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip)          |  384x288   | 70.1 |     trained on 14 datasets      |
|   [RTMW-x](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip)    |  384x288   | 70.2 |     trained on 14 datasets      |

</details>

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
```

## Acknowledgement

Our code is based on these repos:

- [MMPose](https://github.com/open-mmlab/mmpose)
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose)
- [DWPose](https://github.com/IDEA-Research/DWPose/tree/opencv_onnx)
