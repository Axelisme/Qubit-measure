"""Production dispatch, admission and bounded common-frame preparation."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    EnvelopeSpec,
    FamilyRouting,
    ModelSource,
    SecularDrive,
    SecularModel,
)
from zcu_tools.simulate.evolution.kernel import GuardResult
from zcu_tools.simulate.evolution.kernels import REGISTRY

from ._common_frame import batch, model_source
from ._engine import RecordingKernel


def run(engine, source, timeline):
    return engine.evolve(
        source, (timeline,), cancel=Event(), budget=0.01, telemetry=True
    )


def fallback(source, timeline):
    with closing(
        Engine(1, 0, kernels=tuple(k for k in REGISTRY if k.name != "common_frame"))
    ) as engine:
        return run(engine, source, timeline)


def test_shared_constant_coefficients_build_once_and_recheck_apply_only_inputs():
    source = model_source()
    timeline = batch([32, 48, 64, 80], gain=0.3)
    with closing(Engine(1, 1_000_000)) as engine:
        first = run(engine, source, timeline)
        assert first.paths.segments_by_kernel == {"common_frame": 4}
        report = first.paths.routing["common_frame"]
        assert report.prepare_builds == 1
        assert report.admission_rows == 4
        assert report.cache_hits == 0
        # Phase is apply-only, but cannot reuse exact output or row admission.
        changed = replace(
            timeline,
            tracks=tuple(
                replace(t, phase_rad=t.phase_rad + 0.1) for t in timeline.tracks
            ),
        )
        second = run(engine, source, changed)
        assert second.paths.segments_by_kernel == {"common_frame": 4}
        assert second.paths.routing["common_frame"].prepare_builds == 0
        assert second.paths.routing["common_frame"].admission_rows == 4
        assert second.paths.routing["common_frame"].cache_hits == 4
        assert not np.array_equal(first.densities, second.densities)
        repeated = run(engine, source, changed)
        np.testing.assert_array_equal(second.densities, repeated.densities)
        assert repeated.paths.routing["common_frame"].prepare_builds == 0
        assert repeated.paths.routing["common_frame"].admission_rows == 0
        assert repeated.paths.frontend_cpu_s == 0
        assert first.paths.frontend_cpu_s > 0


def test_common_frame_selection_and_results_are_row_order_invariant():
    source = model_source()
    with closing(Engine(1, 0)) as engine:
        forward = run(engine, source, batch([32, 48, 64, 80], gain=0.3))
        reverse = run(engine, source, batch([80, 64, 48, 32], gain=0.3))
    assert (
        forward.paths.segments_by_kernel
        == reverse.paths.segments_by_kernel
        == {"common_frame": 4}
    )
    np.testing.assert_array_equal(forward.densities, reverse.densities[::-1])


@pytest.mark.parametrize("gain,frequency", [(15.0, 0.294), (0.3, 0.7)])
def test_rejected_build_cost_skips_indicator_evaluation_and_preserves_fallback(
    gain, frequency
):
    source = model_source()
    timeline = batch([32, 48, 64, 80], gain=gain, frequency=frequency)
    with closing(Engine(1, 0)) as engine:
        result = run(engine, source, timeline)
    assert (
        getattr(
            result.paths.routing.get("common_frame", FamilyRouting()),
            "admission_rows",
            -1,
        )
        == 0
    )
    original = fallback(source, timeline)
    assert result.paths.segments_by_kernel == original.paths.segments_by_kernel
    np.testing.assert_array_equal(result.densities, original.densities)


def test_attempted_common_frame_numerical_rejection_reuses_complete_fallback():
    source = model_source()
    timeline = batch([32, 48, 64, 80], gain=0.3)
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0.5 + 0.5j, 0.5 - 0.5j] * 81),
        np.arange(162, dtype=float) * 0.5,
    )
    timeline = replace(
        timeline,
        tracks=tuple(replace(t, envelopes=(envelope,)) for t in timeline.tracks),
    )
    with closing(Engine(1, 0)) as engine:
        result = run(engine, source, timeline)
    assert "common_frame" in result.paths.routing
    assert result.paths.routing["common_frame"].admission_rows == 4
    assert result.paths.routing["common_frame"].rejected_rows == 4
    original = fallback(source, timeline)
    assert result.paths.segments_by_kernel == original.paths.segments_by_kernel
    np.testing.assert_array_equal(result.densities, original.densities)


def test_constant_singleton_avoids_unamortized_product_build():
    source = model_source()
    timeline = batch([96], gain=0.6)
    with closing(Engine(1, 0)) as engine:
        result = run(engine, source, timeline)
    original = fallback(source, timeline)
    assert result.paths.segments_by_kernel == original.paths.segments_by_kernel
    np.testing.assert_array_equal(result.densities, original.densities)


@pytest.mark.parametrize("workers,cache_bytes", [(1, 0), (2, 1_000_000)])
def test_rejected_unique_common_keys_preserve_shared_fallback_preparation(
    workers, cache_bytes
):
    # Six levels make the cold common price competitive even without key reuse.
    # Identity kernels isolate fallback sharing from numerical admission policy.
    zero = np.zeros((6, 6), dtype=np.complex128)
    model = SecularModel(
        (0,),
        "shared-fallback",
        np.arange(6, dtype=float) * 0.29,
        np.eye(6, dtype=np.complex128),
        (SecularDrive("a", zero, zero), SecularDrive("b", zero, zero)),
        np.zeros((6, 6)),
        np.zeros((6, 6)),
    )
    source = ModelSource(
        np.diag([1.0, 0, 0, 0, 0, 0]).astype(np.complex128),
        lambda level: replace(model, level=level),
    )
    timeline = batch([80] * 4, gain=0.3)
    timeline = replace(
        timeline,
        tracks=(
            timeline.tracks[0],
            replace(timeline.tracks[1], frequency_ghz=np.linspace(0.291, 0.294, 4)),
        ),
    )

    class RejectedCommon(RecordingKernel):
        guard_builds = True

        def __init__(self):
            super().__init__("common_frame")
            self.admission_rows = 0

        def prepare_keys(self, table, rows, context):
            return table.tone_frequency_ghz[rows, 1].copy()

        def guard_prepared(self, table, rows, context):
            self.admission_rows += len(rows)
            return GuardResult(np.zeros(len(rows), dtype=bool))

    common = RejectedCommon()
    fallback_kernel = RecordingKernel("multitone_rwa")
    with closing(
        Engine(workers, cache_bytes, kernels=(common, fallback_kernel))
    ) as engine:
        result = run(engine, source, timeline)
    assert result.paths.segments_by_kernel == {"multitone_rwa": 4}
    assert common.admission_rows == 4
    assert fallback_kernel.preparations == 1
    np.testing.assert_array_equal(
        result.densities, np.repeat(source.initial_density[None], 4, axis=0)
    )


def test_oversize_products_stream_and_rebuild_without_changing_results():
    source = model_source()
    with closing(Engine(1, 128)) as engine:
        first = run(engine, source, batch([32, 48, 64, 80], gain=0.3))
        second = run(engine, source, batch([40, 56, 72, 88], gain=0.3))
    assert (
        first.paths.segments_by_kernel
        == second.paths.segments_by_kernel
        == {"common_frame": 4}
    )
    assert first.paths.routing["common_frame"].prepare_builds == 1
    assert second.paths.routing["common_frame"].prepare_builds == 1
    assert second.paths.routing["common_frame"].cache_hits == 0
    with closing(Engine(1, 0)) as engine:
        fresh = run(engine, source, batch([40, 56, 72, 88], gain=0.3))
    np.testing.assert_array_equal(second.densities, fresh.densities)
