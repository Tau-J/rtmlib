# rtmlib

rtmlib is a super lightweight library to conduct pose estimation based on [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose) models **WITHOUT** any dependencies like mmcv, mmpose, mmdet, etc.

Basically, rtmlib only requires these dependencies:

- numpy
- opencv-python
- opencv-contrib-python

Optionally, you can use other common backends like pytorch, onnxruntime, tensorrt to accelerate the inference process.

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
- [x] Lightweight models
- [ ] Support alias to choose model
- [ ] Support PoseTracker proposed in RTMPose
- [ ] Support TensorRT backend
- [x] Gradio interface
- [x] Compatible with Controlnet

## Model Zoo

By defaults, rtmlib will automatically download and apply models with the best performance. But you can also specify the model you want to use by passing the `onnx_model` argument.

More models can be found in [RTMPose Model Zoo](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose).

### Detectors

<details open>
<summary><b>Person</b></summary>

Notes:

- Models trained on HumanArt can detect both real human and cartoon characters.
- Models trained on COCO can only detect real human.

|                                                          ONNX Model                                                           | Input Size |       Description       |
| :---------------------------------------------------------------------------------------------------------------------------: | :--------: | :---------------------: |
|                 [YOLOX-l](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing)                 |  640x640   | trained on COCO val2017 |
| [YOLOX-nano](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip) |  416x416   |   trained on HumanArt   |
| [YOLOX-tiny](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip) |  416x416   |   trained on HumanArt   |
|    [YOLOX-s](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_s_8xb8-300e_humanart-3ef259a7.zip)    |  640x640   |   trained on HumanArt   |
|    [YOLOX-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip)    |  640x640   |   trained on HumanArt   |
|    [YOLOX-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_l_8xb8-300e_humanart-ce1d7a62.zip)    |  640x640   |   trained on HumanArt   |
|    [YOLOX-x](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip)    |  640x640   |   trained on HumanArt   |

</details>

### Pose Estimators

<details open>
<summary><b>Body</b></summary>

|                                                                     ONNX Model                                                                      | Input Size |    Description    |
| :-------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :---------------: |
| [RTMPose-t](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip) |  256x192   | Body 17 Keypoints |
| [RTMPose-s](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip) |  256x192   | Body 17 Keypoints |
| [RTMPose-m](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip) |  256x192   | Body 17 Keypoints |
| [RTMPose-l](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip) |  384x288   | Body 17 Keypoints |
| [RTMPose-x](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip) |  384x288   | Body 17 Keypoints |

</details>

<details open>
<summary><b>WholeBody</b></summary>

|                                                                  ONNX Model                                                                   | Input Size |       Description       |
| :-------------------------------------------------------------------------------------------------------------------------------------------: | :--------: | :---------------------: |
| [RTMW-x](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip) |  384x288   | Wholebody 133 Keypoints |

</details>

## Demo

Run `webui.py`:

```shell
# Please make sure you have installed gradio
# pip install gradio

python webui.py
```

![image](https://github.com/Tau-J/rtmlib/assets/13503330/49ef11a1-a1b5-4a20-a2e1-d49f8be6a25d)

Here is also a simple demo to show how to use rtmlib to conduct pose estimation on a single image.

```python
import cv2

from rtmlib import Wholebody, draw_skeleton

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime
img = cv2.imread('./demo.jpg')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
                      backend=backend, device=device)

keypoints, scores = wholebody(img)

# visualize

# if you want to use black background instead of original image,
# img_show = np.zeros(img_show.shape, dtype=np.uint8)

img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)


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
  author={Jiang, Tao},
  year={2023},
  howpublished = {\url{https://github.com/Tau-J/rtmlib}},
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
