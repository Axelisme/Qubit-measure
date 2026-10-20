"""White-noise environments own complete, replayable deferred shot sources."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from zcu_tools.core.simulation import (
    AcquireRequest,
    ReadoutWindow,
    RoundRequest,
    RoundSignals,
)
from zcu_tools.experiment.simulation import white_noise

if TYPE_CHECKING:
    from zcu_tools.program.v2.simulation import IntegratedRound

pytestmark = pytest.mark.sim


def integrated(result: IntegratedRound | RoundSignals) -> IntegratedRound:
    """Narrow one complete integrated delivery for typed numeric assertions."""
    from zcu_tools.program.v2.simulation import IntegratedRound

    if not isinstance(result, IntegratedRound):
        raise AssertionError("integrated environment must deliver statistics and raw")
    return result


def test_white_noise_returns_deferred_full_raw_with_independent_round_totals():
    environment = white_noise(std_adc=3, seed=17)
    try:
        request = AcquireRequest(
            mode="integrated",
            loop_dims=(2, 1200, 3),
            avg_level=1,
            reps=1200,
            readouts=(ReadoutWindow(0, 0, 8, 307.2, (0.0, 0.1)),),
            root_seed=17,
            sequence=0,
            rng_version=environment.soc.binding.rng_version,
        )
        round_request = RoundRequest(
            request, environment.soc.binding.capture_snapshot(), 0
        )
        result = environment.soc.binding.execute(round_request)
        source = getattr(result, "raw", None)
        assert source is not None, (
            "integrated environment must deliver a deferred raw source"
        )
        statistics = integrated(result).statistics
        raw = source.materialize(1200, 1)[0]
        assert raw.shape == (2, 1200, 3, 2, 2)
        assert raw.dtype == np.int64
        assert_allclose(raw.mean(axis=(0, 1, 2, 3)), 0, atol=0.25)
        assert_allclose(raw.var(axis=(0, 1, 2, 3)), 72 + 1 / 12, atol=2.5)
        assert np.max(np.abs(raw.sum(axis=1) - statistics.channels[0].sums)) > 600
        assert_array_equal(source.materialize(1200, 1)[0], raw)
        repeated = integrated(environment.soc.binding.execute(round_request))
        assert_array_equal(
            repeated.statistics.channels[0].sums, statistics.channels[0].sums
        )
        changed = integrated(
            environment.soc.binding.execute(replace(round_request, index=1))
        )
        assert not np.array_equal(changed.raw.materialize(1200, 1)[0], raw)
        with pytest.raises(ValueError, match="reps"):
            source.materialize(1199, 1)
        with pytest.raises(ValueError, match="avg_level|layout"):
            source.materialize(1200, 3)
    finally:
        environment.devices.close_all_devices()
