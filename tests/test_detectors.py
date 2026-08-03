"""Tests for detector-level `det_mode`/`mode` contract consistency.

All three detectors shipped with rtmlib (YOLOX, RTMDet, RFDETR) must:
  * accept `det_mode` ('human' | 'multiclass') with the same semantics;
  * accept the deprecated `mode` kwarg as a backward-compatible alias,
    emitting a `DeprecationWarning`;
  * raise a clear `ValueError` if conflicting `det_mode`/`mode` values are
    both given;
  * expose the resolved value as `self.det_mode` (not `self.mode`), since
    `PoseTracker` and `Custom` both rely on this attribute name.
"""
import warnings
import inspect

import numpy as np
import pytest

from rtmlib.tools.object_detection.rfdetr import RFDETR
from rtmlib.tools.object_detection.rtmdet import RTMDet
from rtmlib.tools.object_detection.yolox import YOLOX

DETECTOR_CLASSES = [YOLOX, RTMDet, RFDETR]


@pytest.mark.parametrize('cls', DETECTOR_CLASSES)
def test_default_det_mode_is_human(cls):
    det = cls(onnx_model='fake.onnx')
    assert det.det_mode == 'human'
    assert not hasattr(det, 'mode'), (
        f'{cls.__name__} should not expose a `mode` attribute; use '
        '`det_mode` (PoseTracker/Custom depend on this).')


@pytest.mark.parametrize('cls', DETECTOR_CLASSES)
def test_deprecated_mode_kwarg_still_works(cls):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        det = cls(onnx_model='fake.onnx', mode='multiclass')
    assert det.det_mode == 'multiclass'
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


@pytest.mark.parametrize('cls', DETECTOR_CLASSES)
def test_conflicting_mode_and_det_mode_raises(cls):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        with pytest.raises(ValueError):
            cls(onnx_model='fake.onnx', det_mode='human', mode='multiclass')


@pytest.mark.parametrize('cls', DETECTOR_CLASSES)
def test_consistent_mode_and_det_mode_does_not_raise(cls):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        det = cls(onnx_model='fake.onnx', det_mode='human', mode='human')
    assert det.det_mode == 'human'


@pytest.mark.parametrize('cls', DETECTOR_CLASSES)
def test_deprecated_mode_is_keyword_only(cls):
    """Regression test: the deprecated `mode` alias must never occupy a
    positional argument slot, otherwise any pre-existing code that calls a
    detector positionally beyond `det_mode` (e.g. passing `nms_thr`/
    `score_thr`/`mean`/`std` positionally) would have its arguments
    silently shifted into `mode`, producing a confusing `ValueError` (or,
    worse, silently wrong behavior) instead of working as before.
    """
    params = list(inspect.signature(cls.__init__).parameters.values())
    mode_param = next(p for p in params if p.name == 'mode')
    assert mode_param.kind == inspect.Parameter.KEYWORD_ONLY, (
        f'{cls.__name__}.__init__`s deprecated `mode` parameter must be '
        'keyword-only.')


class TestPositionalBackwardCompatibility:
    """Locks in the pre-existing positional-argument contract of each
    detector's constructor (as it was before `det_mode` was introduced),
    so that inserting new keyword-only parameters never again shifts these
    positions.
    """

    def test_yolox_positional_args(self):
        det = YOLOX('fake.onnx', (640, 640), 'human', 0.45, 0.7)
        assert (det.det_mode, det.nms_thr, det.score_thr) == (
            'human', 0.45, 0.7)

    def test_rtmdet_positional_args(self):
        det = RTMDet('fake.onnx', (640, 640), 'human', (1, 2, 3), (4, 5, 6))
        assert det.det_mode == 'human'
        assert det.mean == (1, 2, 3)
        assert det.std == (4, 5, 6)

    def test_rfdetr_positional_args(self):
        det = RFDETR('fake.onnx', (576, 576), 'human', 0.3, 300)
        assert det.det_mode == 'human'
        assert det.score_thr == 0.3
        assert det.num_select == 300


class TestRTMDet:

    def test_no_longer_crashes_missing_nms_attrs(self):
        # Regression test: RTMDet used to reference `self.nms_thr` /
        # `self.score_thr` in the NMS-free branch without ever defining
        # them in `__init__`, causing an AttributeError.
        det = RTMDet(onnx_model='fake.onnx')
        assert hasattr(det, 'nms_thr')
        assert hasattr(det, 'score_thr')

    def test_with_nms_branch_behavior_unchanged(self):
        """The baked-in-NMS export path (shape[-1] == 5) is what all
        currently published RTMDet checkpoints (e.g. RTMDet-hand) use.
        Its numeric behavior with default settings must not change."""
        rng = np.random.RandomState(3)
        n = 20
        boxes = rng.rand(1, n, 4)
        scores = rng.rand(1, n)
        outputs = np.concatenate([boxes, scores[..., None]], axis=-1)

        det = RTMDet(onnx_model='fake.onnx')  # det_mode='human' (default)
        result = det.postprocess(outputs.copy(), ratio=1.3)

        expected_mask = scores[0] > 0.3
        expected = boxes[0][expected_mask] / 1.3
        assert np.allclose(result, expected)

    def test_nms_free_branch_human_mode(self):
        model_input_size = (64, 64)
        strides = [8, 16, 32]
        total_anchors = sum(
            (model_input_size[0] // s) * (model_input_size[1] // s)
            for s in strides)
        num_classes = 3
        raw = np.zeros((1, total_anchors, 5 + num_classes), dtype=np.float32)
        # anchor 10 -> class 0 (human)
        raw[0, 10, 4] = 1.0
        raw[0, 10, 5] = 1.0
        raw[0, 10, 2:4] = 0.01
        # anchor 20 -> class 2 (not human)
        raw[0, 20, 4] = 1.0
        raw[0, 20, 7] = 1.0
        raw[0, 20, 2:4] = 0.01

        det = RTMDet(onnx_model='fake.onnx', model_input_size=model_input_size,
                     det_mode='human')
        boxes = det.postprocess(raw.copy(), ratio=1.0)
        assert boxes.shape[0] == 1

    def test_nms_free_branch_multiclass_mode(self):
        model_input_size = (64, 64)
        strides = [8, 16, 32]
        total_anchors = sum(
            (model_input_size[0] // s) * (model_input_size[1] // s)
            for s in strides)
        num_classes = 3
        raw = np.zeros((1, total_anchors, 5 + num_classes), dtype=np.float32)
        raw[0, 10, 4] = 1.0
        raw[0, 10, 5] = 1.0
        raw[0, 10, 2:4] = 0.01
        raw[0, 20, 4] = 1.0
        raw[0, 20, 7] = 1.0
        raw[0, 20, 2:4] = 0.01

        det = RTMDet(onnx_model='fake.onnx', model_input_size=model_input_size,
                     det_mode='multiclass')
        boxes, cls_inds = det.postprocess(raw.copy(), ratio=1.0)
        assert boxes.shape[0] == 2
        assert sorted(cls_inds.tolist()) == [0, 2]

    def test_multiclass_not_supported_with_baked_in_nms(self):
        outputs = np.concatenate(
            [np.random.rand(1, 5, 4), np.random.rand(1, 5, 1)], axis=-1)
        det = RTMDet(onnx_model='fake.onnx', det_mode='multiclass')
        with pytest.raises(NotImplementedError):
            det.postprocess(outputs, ratio=1.0)


class TestCustomDetectorContract:

    def test_all_detectors_accept_det_mode_kwarg(self):
        """`Custom` unconditionally forwards `det_mode=...` to whichever
        detector class the user selects; every shipped detector must
        therefore accept it without raising `TypeError`."""
        for cls in DETECTOR_CLASSES:
            params = inspect.signature(cls.__init__).parameters
            assert 'det_mode' in params, (
                f'{cls.__name__} must accept `det_mode` for `Custom` to '
                'work with it.')
