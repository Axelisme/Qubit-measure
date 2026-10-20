"""Routing reports observe dispatch, cache reuse and numeric transparency."""

import time
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec

from ._engine import RecordingKernel, batch, source, track


def test_family_report_counts_rejections_prepare_hits_and_exact_result_reuse():
    kernels = [
        RecordingKernel("dyson", accepts=False),
        RecordingKernel("t6"),
        RecordingKernel("floquet_period", accepts=False),
    ]
    src = source()
    timelines = replace(batch(4), tracks=(track(4, EnvelopeSpec("constant")),))
    with_engine = Engine(1, 100_000, kernels=kernels)
    try:
        try:
            cold = with_engine.evolve(src, (timelines,), cancel=Event(), telemetry=True)
        except TypeError as error:
            raise AssertionError(
                "evolve must support opt-in routing telemetry"
            ) from error
        routed = cold.paths.routing
        assert routed["dyson"].rows == 0
        assert routed["dyson"].rejected_rows == 4
        assert routed["t6"].rows == 4
        assert routed["t6"].cache_hits == 0
        assert routed["t6"].estimated_cost == 4.0
        assert routed["t6"].estimated_cpu_s is None
        for family in routed.values():
            assert type(family.prepare_cpu_s) is float
            assert type(family.apply_cpu_s) is float
            assert family.actual_cpu_s == family.prepare_cpu_s + family.apply_cpu_s
        changed = replace(timelines, points=np.arange(4, dtype=np.int64) + 10)
        warm = with_engine.evolve(src, (changed,), cancel=Event(), telemetry=True)
        assert warm.paths.routing["t6"].cache_hits == 4
        assert warm.paths.routing["dyson"].rejected_rows == 4
        memo = with_engine.evolve(src, (changed,), cancel=Event(), telemetry=True)
        assert memo.paths.routing["t6"].cache_hits == 4
        assert memo.paths.routing["t6"].actual_cpu_s == 0.0
        assert all(f.rejected_rows == 0 for f in memo.paths.routing.values())
        assert sum(len(call[1]) for call in kernels[1].calls) == 8
        assert np.array_equal(cold.densities, warm.densities)
    finally:
        with_engine.close()


def test_opt_out_avoids_thread_clock_and_keeps_routes_and_density_bitwise(
    monkeypatch: pytest.MonkeyPatch,
):
    src = source()
    timelines = batch(3)
    enabled = Engine(1, 0, kernels=[RecordingKernel("t4")])
    disabled = Engine(1, 0, kernels=[RecordingKernel("t4")])
    try:
        try:
            observed = enabled.evolve(src, (timelines,), cancel=Event(), telemetry=True)
        except TypeError as error:
            raise AssertionError(
                "evolve must support opt-in routing telemetry"
            ) from error

        def forbidden_clock():
            raise AssertionError(
                "disabled telemetry must not read the thread CPU clock"
            )

        monkeypatch.setattr(time, "thread_time", forbidden_clock)
        plain = disabled.evolve(src, (timelines,), cancel=Event(), telemetry=False)
        assert plain.paths.routing == {}
        assert observed.paths.routing["t4"].rows == 3
        assert plain.paths.segments_by_kernel == observed.paths.segments_by_kernel
        assert plain.densities.tobytes() == observed.densities.tobytes()
    finally:
        enabled.close()
        disabled.close()
