"""Shared three-state nearest-center labels for acquisition and analysis."""

import numpy as np
import pytest
from zcu_tools.utils import shot_classification


def test_nearest_centroid3_preserves_shape_and_breaks_ties_in_label_order():
    # Centers (0, 2, 2j): ties g/e at 1, g/f at 1j, all at 1+1j,
    # e/f at 2+2j. Far-away shots still get one of the three labels.
    signals = np.array(
        [[0, 2, 2j, 1], [1j, 1 + 1j, 2 + 2j, -100j]], dtype=np.complex128
    )
    labels = shot_classification.nearest_centroid3(signals, 0j, 2 + 0j, 2j)
    np.testing.assert_array_equal(labels, [[0, 1, 2, 0], [0, 0, 1, 0]])
    assert labels.dtype == np.int64


@pytest.mark.parametrize(
    "signals, centers",
    [
        ([complex(float("nan"), 0)], (0j, 2 + 0j, 2j)),
        ([0j], (0j, 0j, 2j)),
        ([0j], (0j, 2j, 0j)),
        ([0j], (0j, 2j, 2j)),
        ([0j], (complex(float("inf"), 0), 2j, 1j)),
    ],
)
def test_nearest_centroid3_rejects_nonfinite_or_degenerate_input(signals, centers):
    with pytest.raises(ValueError, match="finite|distinct"):
        shot_classification.nearest_centroid3(
            np.asarray(signals, dtype=np.complex128), *centers
        )
