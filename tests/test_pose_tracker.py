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
