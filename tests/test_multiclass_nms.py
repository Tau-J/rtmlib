"""Regression tests for `multiclass_nms`'s `keep` return value.

Historical bug: `multiclass_nms` returned `keep` as an index into the
*score-thresholded subset* of boxes, not into the original input arrays.
Callers that use `keep` to index other per-box arrays sharing the original
ordering (notably `RTMO.postprocess`, which indexes per-box keypoints) would
silently pick up the wrong box/keypoints whenever the surviving detections
were not a sorted-by-score contiguous prefix of the input.
"""
import numpy as np

from rtmlib.tools.object_detection.post_processings import multiclass_nms


def test_keep_is_global_index_single_class():
    """Single-class usage mirrors how `RTMO.postprocess` calls this
    function: scores has a single column."""
    rng = np.random.RandomState(0)
    n = 100
    boxes = rng.rand(n, 4) * 100
    boxes[:, 2:] += boxes[:, :2] + 10
    scores = rng.rand(n) * 0.6  # all below the 0.7 threshold
    winner_idx = 37
    scores[winner_idx] = 0.99
    boxes[winner_idx] = np.array([10, 10, 50, 90])

    dets, keep = multiclass_nms(boxes, scores[:, None], nms_thr=0.45,
                                score_thr=0.7)

    assert dets is not None
    assert keep.tolist() == [winner_idx]
    assert np.allclose(dets[0, :4], boxes[winner_idx])


def test_keep_is_global_index_multiclass():
    rng = np.random.RandomState(0)
    n = 100
    boxes = rng.rand(n, 4) * 100
    boxes[:, 2:] += boxes[:, :2] + 10
    num_classes = 5
    scores = rng.rand(n, num_classes) * 0.6
    scores[10, 2] = 0.95
    scores[80, 4] = 0.9
    scores[80, 0] = 0.85  # same box, high score under two different classes

    dets, keep = multiclass_nms(boxes, scores, nms_thr=0.45, score_thr=0.7)

    assert dets is not None
    for row, idx in zip(dets, keep):
        assert np.allclose(row[:4], boxes[idx])


def test_returns_none_when_nothing_survives():
    boxes = np.random.rand(10, 4) * 100
    scores = np.zeros((10, 1))
    dets, keep = multiclass_nms(boxes, scores, nms_thr=0.45, score_thr=0.7)
    assert dets is None
    assert keep is None


def test_single_candidate():
    boxes = np.array([[0., 0., 10., 10.]])
    scores = np.array([[0.9]])
    dets, keep = multiclass_nms(boxes, scores, nms_thr=0.45, score_thr=0.5)
    assert keep.tolist() == [0]


def test_rtmo_style_keypoints_are_not_misaligned():
    """End-to-end style check mimicking `RTMO.postprocess`: indexing a
    separate `keypoints` array (aligned with the *original* box ordering)
    using `keep` must select the keypoints of the actually-kept box.
    """
    rng = np.random.RandomState(1)
    n, k = 100, 17
    boxes = rng.rand(n, 4) * 100
    boxes[:, 2:] += boxes[:, :2] + 10
    scores = rng.rand(n) * 0.6
    winner_idx = 37
    scores[winner_idx] = 0.95
    boxes[winner_idx] = np.array([10, 10, 50, 90])

    # keypoints[i] is tagged with its own original index `i`.
    keypoints = np.zeros((n, k, 2))
    for i in range(n):
        keypoints[i, :, :] = i

    dets, keep = multiclass_nms(boxes, scores[:, None], nms_thr=0.45,
                                score_thr=0.7)
    kept_keypoints = keypoints[keep]

    assert kept_keypoints.shape[0] == 1
    assert (kept_keypoints[0] == winner_idx).all()
