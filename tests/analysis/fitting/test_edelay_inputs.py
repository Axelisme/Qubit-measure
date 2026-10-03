from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.analysis.fitting.resonance import find_edelay_branch


@pytest.mark.parametrize(
    ("frequency_shape", "signal_shape", "expected_error"),
    [
        (
            (1, 4),
            (4,),
            "electrical-delay branch search expects a one-dimensional frequency "
            "axis and one- or two-dimensional signals",
        ),
        (
            (4,),
            (),
            "electrical-delay branch search expects a one-dimensional frequency "
            "axis and one- or two-dimensional signals",
        ),
        (
            (4,),
            (1, 1, 4),
            "electrical-delay branch search expects a one-dimensional frequency "
            "axis and one- or two-dimensional signals",
        ),
        (
            (4,),
            (3,),
            "electrical-delay branch search frequency and signal lengths must match",
        ),
        (
            (4,),
            (2, 3),
            "electrical-delay branch search frequency and signal lengths must match",
        ),
        (
            (4,),
            (0, 4),
            "electrical-delay branch search requires at least one signal row",
        ),
    ],
    ids=[
        "two-dimensional-frequencies",
        "scalar-signal",
        "three-dimensional-signals",
        "short-trace",
        "short-row-stack",
        "empty-row-stack",
    ],
)
def test_find_edelay_branch_rejects_invalid_shapes(
    frequency_shape: tuple[int, ...],
    signal_shape: tuple[int, ...],
    expected_error: str,
) -> None:
    freqs = np.arange(4, dtype=np.float64).reshape(frequency_shape)
    signals = np.ones(signal_shape, dtype=np.complex128)

    with pytest.raises(ValueError, match=f"^{expected_error}$"):
        find_edelay_branch(freqs, signals)


@pytest.mark.parametrize("row_count", [1, 2], ids=["single-trace", "row-stack"])
def test_find_edelay_branch_recovers_pure_phasor_for_signal_shapes(
    row_count: int,
) -> None:
    freqs = np.asarray([0.0, 0.2, 0.55, 0.8])
    expected_delay = 0.2
    trace = np.exp(-1j * 2.0 * np.pi * freqs * expected_delay)
    signals = trace if row_count == 1 else np.vstack((trace, np.exp(0.37j) * trace))

    assert find_edelay_branch(freqs, signals) == pytest.approx(expected_delay, abs=1e-9)
