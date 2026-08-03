import numpy as np


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy.

    Class-aware version.

    Args:
        boxes (np.ndarray): Boxes in shape (N, 4), xyxy format.
        scores (np.ndarray): Per-class scores in shape (N, num_classes).
        nms_thr (float): IoU threshold for NMS.
        score_thr (float): Score threshold to filter out low-confidence
            boxes before NMS.

    Returns:
        tuple:
        - dets (np.ndarray | None): Kept detections in shape (M, 6),
            formatted as (x1, y1, x2, y2, score, cls_ind). ``None`` if no
            detection survives.
        - keep (np.ndarray | None): Indices into the *original* ``boxes``/
            ``scores`` arrays (axis 0) for each row of ``dets``, i.e.
            ``dets[i, :4] == boxes[keep[i]]``. ``None`` if no detection
            survives. Note that the same original index may appear more
            than once if the corresponding box survives NMS under more
            than one class.
    """
    final_dets = []
    final_keep = []
    num_classes = scores.shape[1]
    all_indices = np.arange(boxes.shape[0])
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            valid_indices = all_indices[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
                # Map the NMS-local `keep` indices (relative to
                # `valid_boxes`) back to indices in the original,
                # unfiltered `boxes`/`scores` arrays. Callers (e.g.
                # RTMO) rely on this to index other per-box arrays
                # (such as keypoints) that share the original ordering.
                final_keep.append(valid_indices[keep])
    if len(final_dets) == 0:
        return None, None
    return np.concatenate(final_dets, 0), np.concatenate(final_keep, 0)
