# Code modified from https://github.com/IDEA-Research/DWPose/blob/opencv_onnx/ControlNet-v1-1-nightly/annotator/dwpose/cv_ox_det.py  # noqa
from typing import List, Tuple

import cv2
import numpy as np

from ..base import BaseTool
from .post_processings import multiclass_nms


class YOLOX(BaseTool):
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (640, 640),
                 mode: str = 'human',
                 nms_thr=0.45,
                 score_thr=0.7,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super().__init__(onnx_model,
                         model_input_size,
                         backend=backend,
                         device=device)
        self.mode = mode
        self.nms_thr = nms_thr
        self.score_thr = score_thr

    def __call__(self, image: np.ndarray):
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)[0]
        results = self.postprocess(outputs, ratio)
        return results

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for YOLOX model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - padded_img (np.ndarray): Preprocessed image.
            - ratio (float): Scale factor applied to the image.
        """
        if img.shape[:2] == tuple(self.model_input_size[:2]):
            padded_img = img.copy()
            ratio = 1.
        else:
            if len(img.shape) == 3:
                padded_img = np.ones(
                    (self.model_input_size[0], self.model_input_size[1], 3),
                    dtype=np.uint8) * 114
            else:
                padded_img = np.ones(self.model_input_size,
                                     dtype=np.uint8) * 114

            ratio = min(self.model_input_size[0] / img.shape[0],
                        self.model_input_size[1] / img.shape[1])
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
            padded_shape = (int(img.shape[0] * ratio),
                            int(img.shape[1] * ratio))
            padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        return padded_img, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for YOLOX model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of YOLOX model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_cls_inds (np.ndarray): Final class IDs.
            - NOT returned: final_scores (np.ndarray): Final scores.
        """

        if outputs.shape[-1] == 4 or outputs.shape[-1] > 5:
            # onnx without nms module

            grids = []
            expanded_strides = []
            strides = [8, 16, 32]

            hsizes = [self.model_input_size[0] // stride for stride in strides]
            wsizes = [self.model_input_size[1] // stride for stride in strides]

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)
            outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

            predictions = outputs[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratio
            dets, keep = multiclass_nms(boxes_xyxy,
                                        scores,
                                        nms_thr=self.nms_thr,
                                        score_thr=self.score_thr)
            if dets is not None:
                pack_dets = (dets[:, :4], dets[:, 4], dets[:, 5])
                final_boxes, final_scores, final_cls_inds = pack_dets
                keep = final_scores > self.nms_thr
                final_boxes = final_boxes[keep]
                final_scores = final_scores[keep]
                final_cls_inds = final_cls_inds[keep].astype(int)
            else:
                final_boxes, final_cls_inds = np.array([]), np.array([])

        elif outputs.shape[-1] == 5:
            # onnx contains nms module

            pack_dets = (outputs[0, :, :4], outputs[0, :, 4])
            final_boxes, final_scores = pack_dets
            final_boxes /= ratio
            isscore = final_scores > 0.3
            isbbox = [i for i in isscore]
            final_boxes = final_boxes[isbbox]

        if self.mode == 'multiclass':
            return final_boxes, final_cls_inds
        elif self.mode == 'human':
            return final_boxes
        else:
            raise NotImplementedError(
                f'Mode must be \'human\' or \'multiclass\': {self.mode} is not supported.'
            )
