import numpy as np


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

    def __init__(self,
                 solution: type,
                 det_frequency: int = 1,
                 mode: str = 'performance',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):

        model = solution(mode=mode,
                         to_openpose=to_openpose,
                         backend=backend,
                         device=device)

        self.det_model = model.det_model
        self.pose_model = model.pose_model

        self.det_frequency = det_frequency
        self.reset()

    def reset(self):
        self.cnt = 0
        self.instance_list = []

    def __call__(self, image: np.ndarray):

        if self.cnt % self.det_frequency == 0:
            bboxes = self.det_model(image)
        else:
            bboxes = self.instance_list

        keypoints, scores = self.pose_model(image, bboxes=bboxes)

        instances = []
        for kpts in keypoints:
            instances.append(pose_to_bbox(kpts))

        self.instance_list = instances
        self.cnt += 1

        return keypoints, scores
