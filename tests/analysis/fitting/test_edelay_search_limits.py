"""Resource limits of electrical-delay branch search."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.analysis.fitting.resonance import find_edelay_branch


def test_find_edelay_branch_fast_fails_before_oversized_radius_overflow() -> None:
    freqs = np.asarray([5000.0, 5000.2, 5000.55, 5001.0])
    signals = np.exp(-1j * 2.0 * np.pi * freqs * 0.2)

    with (
        np.errstate(over="raise"),
        pytest.raises(ValueError, match="candidate resource limit"),
    ):
        find_edelay_branch(
            freqs,
            signals,
            search_radius=np.finfo(np.float64).max,
        )
