from __future__ import annotations

import numpy as np

from zcu_tools.experiment.v2.singleshot.util import correct_populations


def test_none_confusion_matrix_is_noop() -> None:
    """confusion_matrix=None returns the input unchanged (no-op contract)."""
    pops = np.array([0.7, 0.2, 0.1], dtype=np.float64)
    result = correct_populations(pops, None)
    # Same object is returned (deliberate identity no-op).
    assert result is pops
    np.testing.assert_allclose(result, pops)


def test_identity_confusion_matrix_returns_unchanged() -> None:
    """An identity confusion matrix leaves populations unchanged."""
    pops = np.array([0.6, 0.3, 0.1], dtype=np.float64)
    cm = np.eye(3, dtype=np.float64)
    result = correct_populations(pops, cm)
    np.testing.assert_allclose(result, pops)


def test_known_2x2_correction_hand_computed() -> None:
    """A known invertible 2x2 confusion matrix against a hand-computed vector.

    Orientation: populations @ inv(cm), i.e. row-vector convention with the
    state axis as the LAST axis.

    cm = [[0.9, 0.1],
          [0.2, 0.8]]
    inv(cm) = 1/det * [[0.8, -0.1], [-0.2, 0.9]], det = 0.72 - 0.02 = 0.70.
    measured = [0.65, 0.35]
    corrected = measured @ inv(cm)
      col0 = (0.65*0.8 + 0.35*-0.2) / 0.70 = (0.52 - 0.07)/0.70 = 0.45/0.70
      col1 = (0.65*-0.1 + 0.35*0.9) / 0.70 = (-0.065 + 0.315)/0.70 = 0.25/0.70
    """
    cm = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    measured = np.array([0.65, 0.35], dtype=np.float64)

    expected = measured @ np.linalg.inv(cm)
    # Cross-check the hand math too (both within [0, 1] -> no clip effect).
    np.testing.assert_allclose(expected, [0.45 / 0.70, 0.25 / 0.70])

    result = correct_populations(measured, cm)
    np.testing.assert_allclose(result, expected)


def test_clip_negative_and_above_one() -> None:
    """An input that inverts to out-of-range values is clipped into [0, 1]."""
    cm = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    # Pick a measured vector whose unclipped correction goes out of range.
    measured = np.array([0.95, 0.05], dtype=np.float64)

    raw = measured @ np.linalg.inv(cm)
    # Sanity: the raw correction does leave [0, 1] so clipping is exercised.
    assert raw.min() < 0.0 or raw.max() > 1.0

    expected = np.clip(raw, 0.0, 1.0)
    result = correct_populations(measured, cm)

    np.testing.assert_allclose(result, expected)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_batched_input_shape_and_orientation_preserved() -> None:
    """Batched input: state axis is the last axis; batch axis is untouched."""
    cm = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    measured = np.array(
        [
            [0.65, 0.35],
            [0.50, 0.50],
            [0.10, 0.90],
        ],
        dtype=np.float64,
    )

    expected = np.clip(measured @ np.linalg.inv(cm), 0.0, 1.0)
    result = correct_populations(measured, cm)

    assert result.shape == measured.shape
    np.testing.assert_allclose(result, expected)

    # Each row corrected independently (orientation check): row 0 of the
    # batched result equals the single-vector correction of row 0.
    single = correct_populations(measured[0], cm)
    np.testing.assert_allclose(result[0], single)
