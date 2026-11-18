"""Routing reports observe dispatch and numeric transparency."""

import time
from contextlib import closing
from dataclasses import FrozenInstanceError, replace
from operator import setitem
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    EnvelopeSpec,
    PathDiagnostics,
    merge_path_diagnostics,
)

from ._engine import RecordingKernel, batch, source, track


@pytest.mark.parametrize("telemetry", [False, True])
def test_engine_logical_durations_survive_reuse_and_merge(telemetry: bool):
    timelines = replace(batch(3), end_ns=np.array([1.0, 2.0, 4.0]))
    with (
        closing(source()) as src,
        closing(Engine(1, kernels=[RecordingKernel("t4")])) as engine,
    ):
        result = engine.evolve(src, (timelines,), cancel=Event(), telemetry=telemetry)
    assert hasattr(result.paths, "duration_ns_by_kernel"), (
        "Engine must report logical physical duration without CPU telemetry"
    )
    assert result.paths.duration_ns_by_kernel == {"t4": 7.0}
    pytest.raises(TypeError, setitem, result.paths.duration_ns_by_kernel, "t4", 9.0)
    reused = result.paths.for_reuse(telemetry=False).for_reuse(telemetry=True)
    report = reused.routing["t4"]
    assert (report.rows, report.duration_ns, report.cache_hits) == (3, 7.0, 3)
    assert report.prepare_cpu_s == report.apply_cpu_s == report.actual_cpu_s == 0.0
    pytest.raises(FrozenInstanceError, setattr, report, "duration_ns", 9.0)
    merged = merge_path_diagnostics((result.paths, reused))
    assert merged.duration_ns_by_kernel == {"t4": 14.0}
    if telemetry:
        assert result.paths.routing["t4"].duration_ns == 7.0
        assert merged.routing["t4"].duration_ns == 14.0
    else:
        assert result.paths.routing == {}


@pytest.mark.parametrize(
    "durations",
    [{}, {"other": 1.0}, {"t4": -1.0}, {"t4": float("nan")}, {"t4": float("inf")}],
)
def test_diagnostic_duration_rejects_missing_keys_or_invalid_values(
    durations: dict[str, float],
):
    with pytest.raises(ValueError, match="duration"):
        PathDiagnostics(segments_by_kernel={"t4": 1}, duration_ns_by_kernel=durations)


def test_family_report_counts_rejections_and_selected_work():
    kernels = [
        RecordingKernel("dyson", accepts=False),
        RecordingKernel("t6"),
        RecordingKernel("floquet_period", accepts=False),
    ]
    src = source()
    timelines = replace(batch(4), tracks=(track(4, EnvelopeSpec("constant")),))
    with_engine = Engine(1, kernels=kernels)
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
        assert routed["t6"].prepare_key_builds == 1
        assert routed["dyson"].prepare_key_builds == 0
        assert routed["t6"].estimated_cost == 4.0
        assert routed["t6"].estimated_cpu_s is None
        reused = cold.paths.for_reuse(telemetry=True)
        assert reused.routing["t6"].prepare_key_builds == 0
        merged = merge_path_diagnostics((cold.paths, cold.paths, reused))
        assert merged.routing["t6"].prepare_key_builds == 2
        for family in routed.values():
            assert type(family.prepare_cpu_s) is float
            assert type(family.apply_cpu_s) is float
            assert family.actual_cpu_s == family.prepare_cpu_s + family.apply_cpu_s
    finally:
        with_engine.close()


def test_opt_out_avoids_thread_clock_and_keeps_routes_and_density_bitwise(
    monkeypatch: pytest.MonkeyPatch,
):
    src = source()
    timelines = batch(3)
    enabled = Engine(1, kernels=[RecordingKernel("t4")])
    disabled = Engine(1, kernels=[RecordingKernel("t4")])
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
        assert (
            plain.paths.duration_ns_by_kernel
            == observed.paths.duration_ns_by_kernel
            == {"t4": 9.0}
        )
        assert observed.paths.routing["t4"].rows == 3
        assert plain.paths.segments_by_kernel == observed.paths.segments_by_kernel
        assert plain.densities.tobytes() == observed.densities.tobytes()
    finally:
        enabled.close()
        disabled.close()
