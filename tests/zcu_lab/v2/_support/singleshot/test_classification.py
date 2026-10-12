"""Existing program classification parity pending its public owner migration."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from zcu_tools.analysis.singleshot import classify_result
from zcu_tools.program.acquisition import SingleShotMixin


@pytest.mark.parametrize("radius", [0.5, 1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("remove_offset", [False, True])
def test_program_matches_offline(radius: float, remove_offset: bool) -> None:
    signals = np.array([[-1, 1, 0, 0.2, -0.2, 4, -1 + radius]], dtype=np.complex128)
    expected = classify_result(signals, -1, 1, radius)
    assert np.all(sum(mask.astype(int) for mask in expected) == 1)
    assert expected[2][0, 2]  # Exact tie.
    offset = 0.25 if remove_offset else 0.0
    raw = (np.stack((signals.real, signals.imag), axis=-1) + offset) * 10
    program = SimpleNamespace(
        ro_chs={0: {"length": 10}}, soccfg={"readouts": [{"iq_offset": offset}]}
    )
    actual = SingleShotMixin._apply_classification(
        cast(SingleShotMixin, program), [raw], -1, 1, radius, remove_offset
    )
    np.testing.assert_array_equal(actual[0], np.stack(expected[:2], axis=-1))
    assert actual[0].dtype == np.float64
