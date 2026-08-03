"""Tests for `pose_estimation/post_processings.py`, in particular the
`_combine_axis_scores` helper extracted from `get_simcc_maximum` (2D) and
`get_simcc_maximum3d`.

These lock in that:
  * `get_simcc_maximum` still averages the per-axis max responses (as it
    did before the refactor -- this is the value actually shipped in
    RTMPose/RTMW/ViTPose-derived 2D models today).
  * `get_simcc_maximum3d` still takes the min of the two (as it did before
    the refactor -- this is what RTMPose3d/RTMW3D ships today).
  * Neither call site's numeric output changed as a side effect of
    extracting the shared helper.
"""
import numpy as np
import pytest

from rtmlib.tools.pose_estimation.post_processings import (
    _combine_axis_scores, get_simcc_maximum, get_simcc_maximum3d)


def test_combine_axis_scores_mean():
    x = np.array([0.2, 0.8])
    y = np.array([0.6, 0.4])
    result = _combine_axis_scores(x, y, method='mean')
    assert np.allclose(result, [0.4, 0.6])


def test_combine_axis_scores_min():
    x = np.array([0.2, 0.8])
    y = np.array([0.6, 0.4])
    result = _combine_axis_scores(x, y, method='min')
    assert np.allclose(result, [0.2, 0.4])


def test_combine_axis_scores_invalid_method():
    with pytest.raises(ValueError):
        _combine_axis_scores(np.array([0.1]), np.array([0.2]), method='max')


def test_get_simcc_maximum_uses_mean():
    rng = np.random.RandomState(11)
    simcc_x = rng.rand(2, 17, 384).astype(np.float32)
    simcc_y = rng.rand(2, 17, 512).astype(np.float32)

    locs, vals = get_simcc_maximum(simcc_x.copy(), simcc_y.copy())

    max_val_x = simcc_x.reshape(-1, 384).max(axis=1)
    max_val_y = simcc_y.reshape(-1, 512).max(axis=1)
    expected_vals = (0.5 * (max_val_x + max_val_y)).reshape(2, 17)

    assert locs.shape == (2, 17, 2)
    assert vals.shape == (2, 17)
    assert np.allclose(vals, expected_vals)


def test_get_simcc_maximum3d_uses_min():
    rng = np.random.RandomState(11)
    simcc_x = rng.rand(2, 17, 128).astype(np.float32)
    simcc_y = rng.rand(2, 17, 128).astype(np.float32)
    simcc_z = rng.rand(2, 17, 128).astype(np.float32)

    locs, vals = get_simcc_maximum3d(simcc_x.copy(), simcc_y.copy(),
                                     simcc_z.copy())

    max_val_x = simcc_x.reshape(-1, 128).max(axis=1)
    max_val_y = simcc_y.reshape(-1, 128).max(axis=1)
    expected_vals = np.minimum(max_val_x, max_val_y).reshape(2, 17)

    assert locs.shape == (2, 17, 3)
    assert vals.shape == (2, 17)
    assert np.allclose(vals, expected_vals)


def test_get_simcc_maximum_and_3d_disagree_by_design():
    """Documents the known, intentionally-preserved historical
    inconsistency: given identical per-axis max responses, the 2D and 3D
    variants produce different confidence scores. This is not a bug to be
    silently "fixed" by a future refactor without domain input -- see the
    `_combine_axis_scores` docstring."""
    x = np.array([0.9])
    y = np.array([0.1])
    assert not np.isclose(
        _combine_axis_scores(x, y, method='mean'),
        _combine_axis_scores(x, y, method='min'))
