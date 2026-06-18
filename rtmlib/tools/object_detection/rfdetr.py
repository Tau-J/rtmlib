# Code modified from https://github.com/dfki-av/spinepose/blob/main/src/spinepose/tools/object_detection/rfdetr.py  # noqa
from typing import List, Tuple

import cv2
import numpy as np

from ..base import BaseTool


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def bbox_cxcywh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    cx = bboxes[..., 0]
    cy = bboxes[..., 1]
    w = np.clip(bboxes[..., 2], a_min=0.0, a_max=None)
    h = np.clip(bboxes[..., 3], a_min=0.0, a_max=None)
    return np.stack([cx - 0.5 * w,
                     cy - 0.5 * h,
                     cx + 0.5 * w,
                     cy + 0.5 * h], axis=-1)


class RFDETR(BaseTool):
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
                 model_input_size: tuple = (576, 576),
                 mode: str = 'human',
                 score_thr: float = 0.3,
                 num_select: int = 300,
                 mean: tuple = (0.485, 0.456, 0.406),
                 std: tuple = (0.229, 0.224, 0.225),
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super().__init__(onnx_model,
                         model_input_size,
                         mean=mean,
                         std=std,
                         backend=backend,
                         device=device)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.mode = mode
        self.score_thr = score_thr
        self.num_select = num_select

    def __call__(self, image: np.ndarray):
        image, target_sizes = self.preprocess(image)
        outputs = self.inference(image)
        return self.postprocess(outputs, target_sizes)

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for RFDETR model inference.

        Args:
            img (np.ndarray): Input image in shape (H, W, 3).

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - ori_shape (np.ndarray): Original image shape as (H, W).
        """
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError('Expected input image of shape (H, W, 3).')

        ori_shape = np.array(img.shape[:2], dtype=np.float32)

        resized_img = cv2.resize(
            img,
            (int(self.model_input_size[1]), int(self.model_input_size[0])),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        resized_img = (resized_img / 255.0) - self.mean
        resized_img = resized_img / self.std
        return resized_img, ori_shape

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ori_shape: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RFDETR model inference.

        Note: RF-DETR exports include a background class at index 0, so we set its probabilities to -1.0 to exclude it from selection, and map the remaining class IDs to 0-based foreground IDs by subtracting 1.

        Args:
            outputs (List[np.ndarray]): Outputs of RFDETR model.
            ori_shape (np.ndarray): Original image shape as (H, W).

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_cls_inds (np.ndarray): Final class IDs.
            - NOT returned: final_scores (np.ndarray): Final scores.
        """
        boxes, logits = outputs

        probs = sigmoid(logits)
        probs[..., 0] = -1.0

        batch_size, _, num_classes = probs.shape
        batch_indices = np.arange(batch_size)[:, None]

        ori_shape = np.asarray(ori_shape, dtype=np.float32)
        if ori_shape.ndim == 1:
            ori_shape = ori_shape[None, :]
        if ori_shape.shape[0] == 1 and batch_size > 1:
            ori_shape = np.repeat(ori_shape, batch_size, axis=0)
        elif ori_shape.shape[0] != batch_size:
            raise ValueError(
                f'ori_shape batch ({ori_shape.shape[0]}) does not match outputs batch ({batch_size}).'
            )

        flat_probs = probs.reshape(batch_size, -1)
        num_topk = min(self.num_select, flat_probs.shape[1])

        if num_topk > 0:
            topk_unsorted_indices = np.argpartition(flat_probs, -num_topk, axis=1)[
                :, -num_topk:
            ]
            topk_unsorted_scores = np.take_along_axis(
                flat_probs, topk_unsorted_indices, axis=1
            )
            sort_order = np.argsort(-topk_unsorted_scores, axis=1)
            topk_indices = np.take_along_axis(topk_unsorted_indices, sort_order, axis=1)
            scores = np.take_along_axis(topk_unsorted_scores, sort_order, axis=1)

            topk_boxes = topk_indices // num_classes
            labels = topk_indices % num_classes
            mapped_labels = labels - 1

            h = ori_shape[:, 0]
            w = ori_shape[:, 1]
            scale_factors = np.stack([w, h, w, h], axis=1)

            boxes_xyxy = bbox_cxcywh_to_xyxy(boxes)
            boxes_xyxy = boxes_xyxy[batch_indices, topk_boxes]
            boxes_xyxy = boxes_xyxy * scale_factors[:, None, :]

            final_boxes = []
            final_scores = []
            final_cls_inds = []
            for i in range(batch_size):
                keep = scores[i] > self.score_thr  # exclude low-confidence detections
                keep &= labels[i] > 0  # exclude background class
                if self.mode == 'human':
                    keep &= mapped_labels[i] == 0

                final_boxes.append(boxes_xyxy[i][keep].astype(np.float32))
                final_scores.append(scores[i][keep].astype(np.float32))
                final_cls_inds.append(mapped_labels[i][keep].astype(np.int64))

            if len(final_boxes) > 0:
                final_boxes = np.concatenate(final_boxes, axis=0, dtype=np.float32)
                final_scores = np.concatenate(final_scores, axis=0, dtype=np.float32)
                final_cls_inds = np.concatenate(final_cls_inds, axis=0, dtype=np.int64)
            else:
                final_boxes = np.array([], dtype=np.float32)
                final_scores = np.array([], dtype=np.float32)
                final_cls_inds = np.array([], dtype=np.int64)

        else:
            final_boxes = np.array([], dtype=np.float32)
            final_scores = np.array([], dtype=np.float32)
            final_cls_inds = np.array([], dtype=np.int64)

        if self.mode == 'multiclass':
            return final_boxes, final_cls_inds
        elif self.mode == 'human':
            return final_boxes
        else:
            raise NotImplementedError(
                f'Mode must be \'human\' or \'multiclass\': {self.mode} is not supported.'
            )