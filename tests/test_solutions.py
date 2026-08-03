"""Smoke tests: construct every high-level `Solution` (and `Custom` with
every supported `det_class`), then wrap each in `PoseTracker`.

These are intentionally shallow (no real inference is run), but they would
have caught every regression covered in this review:
  * `Custom(det_class='RTMDet' | 'RFDETR', ...)` used to raise `TypeError`
    because `det_mode` was force-passed to detectors that only accepted
    `mode`.
  * `PoseTracker(Body | Wholebody | BodyWithFeet | Wholebody3d | Animal)`
    used to silently end up with `det_model is None`.
"""
import warnings

import pytest

from rtmlib import (Animal, Body, BodyWithFeet, Custom, Hand, PoseTracker,
                    Wholebody, Wholebody3d)

TWO_STAGE_SOLUTIONS = [Body, Wholebody, BodyWithFeet, Hand, Animal,
                       Wholebody3d]


@pytest.mark.parametrize('solution_cls', TWO_STAGE_SOLUTIONS)
def test_two_stage_solution_constructs_with_detector(solution_cls):
    solution = solution_cls()
    assert getattr(solution, 'det_model', None) is not None


@pytest.mark.parametrize('solution_cls', TWO_STAGE_SOLUTIONS)
def test_pose_tracker_keeps_detector_for_two_stage_solutions(solution_cls):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tracker = PoseTracker(solution_cls, det_frequency=5)
    assert tracker.det_model is not None, (
        f'PoseTracker({solution_cls.__name__}) unexpectedly has no '
        'detector -- this is the exact regression this test suite guards '
        'against.')


@pytest.mark.parametrize('det_class', ['YOLOX', 'RTMDet', 'RFDETR'])
def test_custom_constructs_with_every_detector_class(det_class):
    custom = Custom(
        det_class=det_class,
        det='fake_det.onnx',
        pose_class='RTMPose',
        pose='fake_pose.onnx',
    )
    assert custom.det_model is not None
    assert custom.det_model.det_mode == 'human'


def test_custom_one_stage_has_no_detector():
    custom = Custom(pose_class='RTMO', pose='fake_rtmo.onnx',
                    pose_input_size=(640, 640))
    assert custom.one_stage is True
    assert not hasattr(custom, 'det_model')


def test_custom_accepts_kwargs_style_detector_class():
    """Regression test: `Custom`'s defensive `det_mode` support check used
    to look only for a literal `det_mode` parameter name in the detector's
    signature, which would incorrectly reject a perfectly valid detector
    class that accepts `det_mode` via `**kwargs`."""
    import rtmlib

    class _KwargsDetector:

        def __init__(self, onnx_model, model_input_size=(640, 640), **kwargs):
            self.det_mode = kwargs.get('det_mode', 'human')

    rtmlib._KwargsDetector = _KwargsDetector
    try:
        custom = Custom(det_class='_KwargsDetector', det='fake.onnx',
                        pose_class='RTMPose', pose='fake.onnx')
        assert custom.det_model.det_mode == 'human'
    finally:
        del rtmlib._KwargsDetector


def test_custom_rejects_detector_without_det_mode_support():
    import rtmlib

    class _NoDetModeDetector:

        def __init__(self, onnx_model, model_input_size=(640, 640), backend='onnxruntime', device='cpu'):
            pass

    rtmlib._NoDetModeDetector = _NoDetModeDetector
    try:
        with pytest.raises(TypeError):
            Custom(det_class='_NoDetModeDetector', det='fake.onnx', pose_class='RTMPose', pose='fake.onnx')
    finally:
        del rtmlib._NoDetModeDetector
