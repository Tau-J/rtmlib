"""Regression tests for `PoseTracker`'s detector-attribute contract.

Historical bug: `PoseTracker.__init__` read `self.det_model.mode`, which
only ever existed on `RTMDet`/`RFDETR`. After YOLOX's `mode` attribute was
renamed to `det_mode` (to disambiguate from the solution-level
`mode='balanced'/...`), every YOLOX-based high-level solution (`Body`,
`Wholebody`, `BodyWithFeet`, `Wholebody3d`, `Animal`) started raising an
`AttributeError` here -- which a broad `except Exception` silently
swallowed, downgrading `PoseTracker` to "no detector" mode without any
visible error. This file locks in the fix:
  * a detector that exposes `det_mode` is used as expected;
  * the *expected* "no detector" case (one-stage models such as RTMO, which
    never set `self.det_model` at all) still works silently, as designed;
  * a detector that has neither `det_mode` nor is absent (a genuine
    contract violation) now raises loudly instead of degrading silently.
"""
import numpy as np
import pytest

from rtmlib.tools.solution.pose_tracker import PoseTracker


class _FakePoseModel:
    pass


class _WorkingDetector:
    """Stands in for YOLOX/RTMDet/RFDETR: exposes `det_mode`."""

    def __init__(self):
        self.det_mode = 'human'


class _BrokenDetector:
    """A detector-shaped object that forgot to set `det_mode` -- this is
    exactly the situation the original bug hit (YOLOX exposing `det_mode`
    while `PoseTracker` looked for `.mode`)."""


class _TwoStageSolution:
    """Mimics Body/Wholebody/etc. in their default (two-stage) mode."""

    def __init__(self, mode=None, to_openpose=False, backend=None, device=None):
        self.det_model = _WorkingDetector()
        self.pose_model = _FakePoseModel()


class _OneStageSolution:
    """Mimics Body/Custom when configured with a one-stage model (e.g.
    RTMO): no `det_model` attribute is ever set, by design."""

    def __init__(self, mode=None, to_openpose=False, backend=None, device=None):
        self.pose_model = _FakePoseModel()


class _BrokenSolution:
    """A solution whose detector is missing the `det_mode` contract --
    this must be treated as a bug, not silently downgraded."""

    def __init__(self, mode=None, to_openpose=False, backend=None, device=None):
        self.det_model = _BrokenDetector()
        self.pose_model = _FakePoseModel()


def test_two_stage_solution_keeps_its_detector():
    tracker = PoseTracker(_TwoStageSolution)
    assert tracker.det_model is not None
    assert tracker.det_mode == 'human'


def test_one_stage_solution_has_no_detector_and_does_not_raise():
    tracker = PoseTracker(_OneStageSolution)
    assert tracker.det_model is None
    assert tracker.det_mode is None
    assert tracker.det_categories is None


def test_detector_missing_det_mode_raises_instead_of_silently_disabling():
    with pytest.raises(AttributeError):
        PoseTracker(_BrokenSolution)


def test_det_categories_forces_multiclass():

    class _WithCategories(_TwoStageSolution):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.det_categories = [0, 16]

    tracker = PoseTracker(_WithCategories)
    assert tracker.det_mode == 'multiclass'
    assert tracker.det_categories == [0, 16]


# --------------------------------------------------------------------------
# Track id <-> output alignment (issue #74).
#
# Each scripted frame lists the boxes the fake detector returns. The fake
# pose model puts one keypoint on each box corner, so `pose_to_bbox`
# recovers the box and the x of the first keypoint identifies the person.
# --------------------------------------------------------------------------
A = [0, 0, 100, 200]
B = [300, 0, 400, 200]
C = [600, 0, 700, 200]
TINY = [900, 0, 905, 5]  # area below PoseTracker.MIN_AREA


def _corners(box):
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=float)


def _poses(boxes):
    keypoints = np.array([_corners(b) for b in boxes]).reshape(-1, 4, 2)
    return keypoints, np.ones(keypoints.shape[:2])


class _ScriptedDetector:

    def __init__(self, frames):
        self.det_mode = 'human'
        self.frames = iter(frames)

    def __call__(self, image):
        return list(next(self.frames))


class _BoxPoseModel:

    def __call__(self, image, bboxes):
        return _poses(bboxes)


class RTMPose3d:
    """Named like the real class, because PoseTracker branches on the type
    name. Its 3D keypoints are in a different space than the image-space 2D
    keypoints it returns last, as in the real model."""

    def __call__(self, image, bboxes):
        keypoints2d, scores = _poses(bboxes)
        keypoints = np.concatenate(
            [keypoints2d / 10,
             np.zeros(keypoints2d.shape[:2] + (1, ))],
            axis=-1)
        return keypoints, scores, keypoints.copy(), keypoints2d


def _run(frames, pose_model=None, n_frames=None, **kwargs):
    """Track `n_frames` frames (default: one per detector call in `frames`).

    Returns (track ids, first-keypoint x of each pose, raw outputs) per frame.
    """

    class _Solution:

        def __init__(self, **_):
            self.det_model = _ScriptedDetector(frames)
            self.pose_model = pose_model or _BoxPoseModel()

    tracker = PoseTracker(_Solution, **kwargs)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    results = []
    for _ in range(n_frames or len(frames)):
        outputs = tracker(image)
        keypoints = outputs[-1] if len(outputs) == 4 else outputs[0]
        results.append((list(tracker.track_ids_last_frame),
                        [int(k[0][0]) for k in keypoints], outputs))
    return results


def test_track_ids_survive_a_person_leaving_and_entering():
    results = _run([[A, B, C], [A, B], [A, B, C], [A, B, C]])
    assert [ids for ids, _, _ in results] == [[0, 1, 2], [0, 1], [0, 1, 3],
                                              [0, 1, 3]]
    assert results[-1][1] == [0, 300, 600]


def test_untracked_tiny_pose_is_dropped_and_ids_stay_aligned():
    ids, xs, (keypoints, scores) = _run([[A, B], [TINY, A, B]])[-1]
    assert ids == [0, 1]
    assert xs == [0, 300]
    assert len(keypoints) == len(scores) == 2


def test_ids_are_not_reused_after_an_empty_frame():
    results = _run([[A, B], [], [B], [A, B]])
    ids, xs, (keypoints, scores) = results[1]
    assert ids == [] and keypoints.shape == (0, 4, 2)
    assert scores.shape == (0, 4)
    assert results[2][0] == [2]
    assert results[3][0] == [3, 2]
    assert results[3][1] == [0, 300]


def test_crossing_people_keep_their_ids_by_overlap():
    a_moved = [40, 0, 140, 200]
    ids, xs, _ = _run([[A, B], [B, a_moved]])[-1]
    assert ids == [1, 0]
    assert xs == [300, 40]


def test_ids_persist_across_frames_without_detection():
    # The detector runs on frames 0 and 2; frame 1 reuses the tracked boxes.
    results = _run([[A, B], [A, TINY, B]], n_frames=3, det_frequency=2)
    assert [ids for ids, _, _ in results] == [[0, 1], [0, 1], [0, 1]]
    assert results[-1][1] == [0, 300]


def test_tracking_off_returns_every_pose_in_detection_order():
    ids, xs, (keypoints, _) = _run([[A, B], [TINY, B, A]], tracking=False)[-1]
    assert xs == [900, 300, 0]
    assert len(keypoints) == 3


def test_rtmpose3d_tracks_with_2d_keypoints_and_keeps_outputs_aligned():
    results = _run([[A, B], [TINY, B, A]], pose_model=RTMPose3d())
    ids, xs, outputs = results[-1]
    assert ids == [1, 0]
    assert xs == [300, 0]
    assert len(outputs) == 4
    assert len({len(o) for o in outputs}) == 1
