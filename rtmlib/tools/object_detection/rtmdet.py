import warnings
from typing import List, Tuple

import cv2
import numpy as np

from ..base import BaseTool
from .post_processings import multiclass_nms


class RTMDet(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (640, 640),
                 det_mode: str = None,
                 mean: tuple = (103.5300, 116.2800, 123.6750),
                 std: tuple = (57.3750, 57.1200, 58.3950),
                 backend: str = 'onnxruntime',
                 device: str = 'cpu',
                 *,
                 mode: str = None,
                 nms_thr: float = 0.45,
                 score_thr: float = 0.3):
        super().__init__(onnx_model,
                         model_input_size,
                         mean=mean,
                         std=std,
                         backend=backend,
                         device=device)

        # `mode` is deprecated in favor of `det_mode`, for consistency
        # with YOLOX/RFDETR and to avoid confusion with the solution-level
        # `mode='balanced'/'performance'/'lightweight'` argument. Kept
        # keyword-only so it can never accidentally capture a positional
        # argument meant for `mean`/`std`/`backend`/`device` -- those kept
        # their exact original positions (3rd/4th/5th/6th) for backward
        # compatibility with pre-existing positional call sites.
        if mode is not None:
            warnings.warn(
                '`mode` is deprecated for RTMDet, please use `det_mode` '
                'instead. Support for `mode` will be removed in a future '
                'release.', DeprecationWarning, stacklevel=2)
            if det_mode is not None and det_mode != mode:
                raise ValueError(
                    f'Conflicting values for `det_mode` ({det_mode!r}) and '
                    f'the deprecated `mode` ({mode!r}). Please only specify '
                    '`det_mode`.')
            det_mode = mode

        self.det_mode = det_mode if det_mode is not None else 'human'
        self.nms_thr = nms_thr
        self.score_thr = score_thr

    def __call__(self, image: np.ndarray):
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)[0]
        results = self.postprocess(outputs, ratio)
        return results

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for RTMPose model inference.

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

        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            padded_img = (padded_img - self.mean) / self.std

        return padded_img, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMDet model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMDet model.
            ratio (float): Ratio of preprocessing.

        Returns:
            np.ndarray | tuple:
            - If ``det_mode == 'human'``: final_boxes (np.ndarray).
            - If ``det_mode == 'multiclass'``: a tuple of
              (final_boxes, final_cls_inds).
        """
        if self.det_mode not in ('human', 'multiclass'):
            raise NotImplementedError(
                f'det_mode must be \'human\' or \'multiclass\': '
                f'{self.det_mode} is not supported.')

        # NOTE: matches the condition used by YOLOX.postprocess. The actual
        # NMS-free format is (cx, cy, w, h, obj_score, cls_score...), i.e.
        # `shape[-1] > 5`; `shape[-1] == 4` is kept only for parity/defensive
        # symmetry with YOLOX and is not expected to occur in practice.
        if outputs.shape[-1] == 4 or outputs.shape[-1] > 5:
            # onnx without nms module: raw per-class scores are available,
            # so both 'human' and 'multiclass' det_mode are supported here.

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
                # `multiclass_nms` already filters by `self.score_thr`, so
                # no extra score filtering is needed here.
                final_boxes, final_scores, final_cls_inds = (
                    dets[:, :4], dets[:, 4], dets[:, 5].astype(int))
                if self.det_mode == 'human':
                    final_boxes = final_boxes[final_cls_inds == 0]
            else:
                final_boxes = np.empty((0, 4))
                final_cls_inds = np.empty((0, ), dtype=int)

        elif outputs.shape[-1] == 5:
            # onnx contains a baked-in nms module: only box + score are
            # exported (no per-box class id), so 'multiclass' det_mode
            # cannot be supported for this export format.
            if self.det_mode == 'multiclass':
                raise NotImplementedError(
                    'det_mode=\'multiclass\' requires per-class scores, '
                    'but this RTMDet onnx export already contains a '
                    'baked-in NMS module and only exposes (box, score) '
                    'per detection, with no class id. Please export the '
                    'model without a baked-in NMS module to use '
                    'multiclass detection.')

            final_boxes, final_scores = outputs[0, :, :4], outputs[0, :, 4]
            final_boxes = final_boxes / ratio
            isscore = final_scores > self.score_thr
            final_boxes = final_boxes[isscore]
            final_cls_inds = np.zeros((final_boxes.shape[0], ), dtype=int)

        else:
            raise ValueError(
                f'Unexpected RTMDet output shape {outputs.shape}: last '
                'dimension must be 4 (no baked-in NMS) or 5 (baked-in NMS).')

        if self.det_mode == 'multiclass':
            return final_boxes, final_cls_inds
        return final_boxes
