'''
Example:

import cv2
from functools import partial
from rtmlib import PoseTracker, Wholebody, Custom, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

cap = cv2.VideoCapture('./demo.mp4')

pose_tracker = PoseTracker(Wholebody,
                        det_frequency=10,  # detect every 10 frames
                        to_openpose=openpose_skeleton,
                        backend=backend, device=device)


# # Initialized slightly differently for Custom solution:
# custom = partial(Custom,
#                 to_openpose=openpose_skeleton,
#                 pose_class='RTMO',
#                 pose='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip', # noqa
#                 pose_input_size=(640,640),
#                 backend=backend,
#                 device=device)
# # or
# custom = partial(
#             Custom,
#             to_openpose=openpose_skeleton,
#             det_class='YOLOX',
#             det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip', # noqa
#             det_input_size=(640, 640),
#             pose_class='RTMPose',
#             pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip', # noqa
#             pose_input_size=(192, 256),
#             backend=backend,
#             device=device)
# # then
# pose_tracker = PoseTracker(custom,
#                         det_frequency=10,
#                         to_openpose=openpose_skeleton,
#                         backend=backend, device=device)


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

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)
'''
import warnings

import numpy as np


def compute_iou(bboxA, bboxB):
    """Compute the Intersection over Union (IoU) between two boxes .

    Args:
        bboxA (list): The first bbox info (left, top, right, bottom, score).
        bboxB (list): The second bbox info (left, top, right, bottom, score).

    Returns:
        float: The IoU value.
    """

    x1 = max(bboxA[0], bboxB[0])
    y1 = max(bboxA[1], bboxB[1])
    x2 = min(bboxA[2], bboxB[2])
    y2 = min(bboxA[3], bboxB[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    union_area = float(bboxA_area + bboxB_area - inter_area)
    if union_area == 0:
        union_area = 1e-5
        warnings.warn('union_area=0 is unexpected')

    iou = inter_area / union_area

    return iou


def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    """Get bounding box from keypoints.

    Args:
        keypoints (np.ndarray): Keypoints of person.
        expansion (float): Expansion ratio of bounding box.

    Returns:
        np.ndarray: Bounding box of person.
    """
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    bbox = np.concatenate([
        center - (center - bbox[:2]) * expansion,
        center + (bbox[2:] - center) * expansion
    ])
    return bbox


class PoseTracker:
    """Pose tracker for pose estimation.

    Args:
        solution (type): rtmlib solutions, e.g. Wholebody, Body, Custom, etc.
        det_frequency (int): Frequency of object detection.
        mode (str, optional): 'performance', 'lightweight', or 'balanced'.
            If ``None`` (default), the wrapped solution's own default mode
            is used instead of being overridden. Not every solution
            supports every mode (e.g. ``Hand`` only supports
            ``'lightweight'``), so only pass this if you intend to
            override the solution's default.
        to_openpose (bool): Whether to use openpose-style skeleton.
        backend (str): Backend of pose estimation model.
        device (str): Device of pose estimation model.
    """
    MIN_AREA = 1000

    def __init__(self,
                 solution: type,
                 det_frequency: int = 1,
                 tracking: bool = True,
                 tracking_thr: float = 0.3,
                 mode: str = None,
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):

        # Only forward `mode` if the caller explicitly provided one.
        # Solutions do not share the same set of valid `mode` values (e.g.
        # `Hand` only supports `mode='lightweight'`, while `Body`/
        # `Wholebody`/etc. default to `mode='balanced'`), so unconditionally
        # forcing a single default here would override -- and potentially
        # break -- a solution's own, valid default.
        solution_kwargs = dict(to_openpose=to_openpose,
                               backend=backend,
                               device=device)
        if mode is not None:
            solution_kwargs['mode'] = mode
        model = solution(**solution_kwargs)

        det_model = getattr(model, 'det_model', None)
        if det_model is None:
            # Expected for one-stage algorithms (e.g. RTMO / Custom without
            # `det_class`), which have no separate detector by design.
            self.det_model = None
            self.det_mode = None
            self.det_categories = None
        else:
            if not hasattr(det_model, 'det_mode'):
                # This is *not* the expected "one-stage, no detector" case:
                # a detector instance exists but doesn't expose the
                # `det_mode` contract that every rtmlib detector (YOLOX /
                # RTMDet / RFDETR) is expected to provide. Silently
                # disabling detection here would make PoseTracker fall back
                # to running pose estimation on the full image every frame,
                # which quietly breaks multi-person tracking without any
                # visible error. Fail loudly instead.
                raise AttributeError(
                    f'{type(det_model).__name__} (used as the detector of '
                    f'{solution}) has no `det_mode` attribute. PoseTracker '
                    'requires detectors that expose `det_mode` (\'human\' '
                    'or \'multiclass\'), as YOLOX/RTMDet/RFDETR do. If you '
                    'are using a custom detector class, make sure it sets '
                    '`self.det_mode`.')

            self.det_model = det_model
            self.det_mode = det_model.det_mode
            if hasattr(model, 'det_categories') and model.det_categories:
                self.det_mode = 'multiclass'
                self.det_categories = model.det_categories
            else:
                self.det_categories = None

        self.pose_model = model.pose_model

        self.det_frequency = det_frequency
        self.tracking = tracking
        self.tracking_thr = tracking_thr
        self.reset()

        if self.tracking:
            print('Tracking is on, you can get higher FPS by turning it off:'
                  '`PoseTracker(tracking=False)`')

    def reset(self):
        """Reset pose tracker."""
        self.frame_cnt = 0
        self.next_id = 0
        self.bboxes_last_frame = []
        self.track_ids_last_frame = []

    def __call__(self, image: np.ndarray):

        pose_model_name = type(self.pose_model).__name__

        if self.det_model:  # top-down algorithm, e.g. rtmpose
            if self.frame_cnt % self.det_frequency == 0:
                try:
                    if self.det_categories or self.det_mode == 'multiclass':
                        if self.det_categories:
                            bboxes, classes = self.det_model(image)
                            bboxes = [
                                bbox for bbox, cls in zip(bboxes, classes)
                                if cls in self.det_categories
                            ]
                        else:
                            bboxes, _ = self.det_model(image)
                    else:
                        bboxes = self.det_model(image)
                except:  # noqa
                    return [], []
            else:
                bboxes = self.bboxes_last_frame

            if pose_model_name == 'RTMPose3d':
                keypoints, scores, keypoints_simcc, keypoints2d = self.pose_model(
                    image, bboxes=bboxes)
            else:
                keypoints, scores = self.pose_model(image, bboxes=bboxes)

        else:  # one-stage algorithm, e.g. rtmo
            keypoints, scores = self.pose_model(image)

        if not self.tracking and self.det_frequency != 1:
            # without tracking
            bboxes_current_frame = []
            if pose_model_name == 'RTMPose3d':
                for kpts in keypoints2d:
                    bbox = pose_to_bbox(kpts)
                    bboxes_current_frame.append(bbox)
            else:
                for kpts in keypoints:
                    bbox = pose_to_bbox(kpts)
                    bboxes_current_frame.append(bbox)

        else:
            # with tracking
            if len(self.track_ids_last_frame) == 0:
                self.next_id = len(self.bboxes_last_frame)
                self.track_ids_last_frame = list(range(self.next_id))

            bboxes_current_frame = []
            track_ids_current_frame = []
            for kpts in keypoints:
                bbox = pose_to_bbox(kpts)

                track_id, _ = self.track_by_iou(bbox)

                if track_id > -1:
                    track_ids_current_frame.append(track_id)
                    bboxes_current_frame.append(bbox)

            self.track_ids_last_frame = track_ids_current_frame
            # reorder keypoints, scores according to track_id
            try:
                keypoints = np.array(
                    [keypoints[i] for i in self.track_ids_last_frame])
                scores = np.array(
                    [scores[i] for i in self.track_ids_last_frame])
            except:  # noqa
                # in case track_ids_current_frame is empty
                return keypoints, scores,

        self.bboxes_last_frame = bboxes_current_frame
        self.frame_cnt += 1

        if pose_model_name == 'RTMPose3d':
            return keypoints, scores, keypoints_simcc, keypoints2d

        return keypoints, scores

    def track_by_iou(self, bbox):
        """Get track id using IoU tracking greedily.

        Args:
            bbox (list): The bbox info (left, top, right, bottom, score).
            next_id (int): The next track id.

        Returns:
            track_id (int): The track id.
            match_result (list): The matched bbox.
            next_id (int): The updated next track id.
        """

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        max_iou_score = -1
        max_index = -1
        match_result = None
        for index, each_bbox in enumerate(self.bboxes_last_frame):

            iou_score = compute_iou(bbox, each_bbox)
            if iou_score > max_iou_score:
                max_iou_score = iou_score
                max_index = index

        if max_iou_score > self.tracking_thr:
            # if the bbox has a match and the IoU is larger than threshold
            track_id = self.track_ids_last_frame.pop(max_index)
            match_result = self.bboxes_last_frame.pop(max_index)

        elif area >= self.MIN_AREA:
            # no match, but the bbox is large enough,
            # assign a new track id
            track_id = self.next_id
            self.next_id += 1

        else:
            # if the bbox is too small, ignore it
            track_id = -1

        return track_id, match_result
