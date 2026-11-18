"""Production dispatch, admission and bounded common-frame preparation."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    Engine,
    EnvelopeSpec,
    FamilyRouting,
    ModelSource,
    SecularDrive,
    SecularModel,
    read_capture_metadata,
)
from zcu_tools.simulate.evolution.kernel import GuardResult

from ._common_frame import batch, model_source
from ._engine import RecordingKernel


def run(engine, source, timeline):
    return engine.evolve(
        source, (timeline,), cancel=Event(), budget=0.01, telemetry=True
    )


def fallback(source, timeline, *, budget=0.01):
    with closing(
        Engine(1, families=("multitone_rwa", "multitone_gate", "floquet", "dop853"))
    ) as engine:
        return engine.evolve(
            source, (timeline,), cancel=Event(), budget=budget, telemetry=True
        )


def test_widely_spaced_carriers_are_pinnable_but_default_skips_unprofitable_work():
    with closing(model_source()) as source:
        timeline = batch([16], gain=0.03, frequency=0.39)
        with closing(Engine(1, families=("common_frame",))) as engine:
            pinned = run(engine, source, timeline)
        assert pinned.paths.segments_by_kernel == {"common_frame": 1}
        assert pinned.paths.routing["common_frame"].rejected_rows == 0

        with closing(Engine(1)) as engine:
            default = run(engine, source, timeline)
        report = default.paths.routing.get("common_frame", FamilyRouting())
        assert report.admission_rows == 0
        original = fallback(source, timeline)
        assert default.paths.segments_by_kernel == original.paths.segments_by_kernel
        np.testing.assert_array_equal(default.densities, original.densities)
        np.testing.assert_allclose(
            pinned.densities, original.densities, atol=2e-3, rtol=0
        )


def test_shared_constant_coefficients_build_once_for_all_rows():
    source = model_source()
    timeline = batch([32, 48, 64, 80], gain=0.3)
    with closing(Engine(1)) as engine:
        first = run(engine, source, timeline)
        assert first.paths.segments_by_kernel == {"common_frame": 4}
        report = first.paths.routing["common_frame"]
        assert report.prepare_builds == 1
        assert report.admission_rows == 4
        assert first.paths.frontend_cpu_s > 0


@pytest.mark.parametrize("split_phase", [False, True])
def test_sampled_price_hands_the_adopted_common_frame_to_apply(split_phase):
    # A dense grid makes the serial RWA probe expensive, but the smooth
    # envelope and nearby carriers remain eligible for common frame.
    timeline = batch([80.0] * 4, gain=0.2)
    knots = np.linspace(0.0, 80.0, 1200)
    envelope = EnvelopeSpec(
        "sampled", np.linspace(0.8, 1.0, len(knots)).astype(complex), knots
    )
    tracks = tuple(replace(t, envelopes=(envelope,)) for t in timeline.tracks)
    if split_phase:
        tracks = (
            replace(tracks[0], phase_rad=np.array([0.37, 0.37, 0.51, 0.51])),
            tracks[1],
        )
    timeline = replace(timeline, tracks=tracks)
    with closing(model_source()) as source, closing(Engine(1, backend="cpu")) as engine:
        result = engine.evolve(source, (timeline,), telemetry=True)
    assert result.paths.segments_by_kernel == {"common_frame": 4}
    assert result.paths.routing["common_frame"].prepare_builds == 1
    assert result.paths.routing["common_frame"].admission_rows == 4


@pytest.mark.parametrize("far,sampled", [(False, False), (True, False), (True, True)])
def test_constant_block_rwa_avoids_unprofitable_unique_common_builds(far, sampled):
    # Two independent three-state blocks have retained dimension 18, above the
    # small-graph shortcut, but cheap constant RWA work even for long rows.
    x = np.zeros((6, 6), dtype=np.complex128)
    for i in (0, 1, 3, 4):
        x[i, i + 1] = x[i + 1, i] = 0.005
    model = SecularModel(
        (0,),
        "constant-blocks",
        np.array([0.0, 0.29, 0.58, 1.1, 1.39, 1.68]),
        np.eye(6, dtype=np.complex128),
        (
            SecularDrive("a", x, np.zeros_like(x)),
            SecularDrive("b", x, np.zeros_like(x)),
        ),
        np.zeros((6, 6)),
        np.zeros((6, 6)),
    )
    source = ModelSource(
        np.diag([1.0, 0, 0, 0, 0, 0]).astype(np.complex128),
        lambda level: replace(model, level=level),
    )
    timeline = batch([1000] * 4, gain=0.1)
    if sampled:
        envelope = EnvelopeSpec(
            "sampled",
            np.array([0.8, 1.0], dtype=np.complex128),
            np.array([0.0, 1000.0]),
        )
        timeline = replace(
            timeline,
            tracks=tuple(replace(t, envelopes=(envelope,)) for t in timeline.tracks),
        )
    timeline = replace(
        timeline,
        tracks=(
            timeline.tracks[0],
            replace(
                timeline.tracks[1],
                frequency_ghz=np.linspace(0.7, 0.703, 4)
                if far
                else np.linspace(0.291, 0.294, 4),
            ),
        ),
    )
    with closing(Engine(1)) as engine:
        result = engine.evolve(
            source, (timeline,), cancel=Event(), budget=1e-3, telemetry=True
        )
    if not sampled:
        assert result.paths.segments_by_kernel == {"multitone_rwa": 4}
    report = result.paths.routing.get("common_frame", FamilyRouting())
    assert report.prepare_builds == report.admission_rows == 0
    original = fallback(source, timeline, budget=1e-3)
    assert result.paths.segments_by_kernel == original.paths.segments_by_kernel
    np.testing.assert_array_equal(result.densities, original.densities)

    # A genuinely amortized coefficient group must still be eligible. This is
    # cold selection with no resident cache, not a cache-warmth exception.
    with closing(Engine(1)) as engine:
        reused = run(engine, model_source(), batch([32] * 24, gain=0.3))
    assert reused.paths.segments_by_kernel == {"common_frame": 24}
    assert reused.paths.routing["common_frame"].prepare_builds == 1
    assert reused.paths.routing["common_frame"].admission_rows == 24


def test_shared_sampled_prefix_does_not_borrow_another_points_constant_route():
    # Point zero has a cheap constant pair. The other points share a sampled
    # pair up to the final pulse, which has different gains on the two points.
    plateau = batch([12.0, 12.0, 12.0], gain=0.03)
    sampled = batch([192.0] * 3, gain=0.2, frequency=0.295)
    envelope = EnvelopeSpec(
        "sampled", np.array([0.8, 1.0, 0.9], complex), np.array([0.0, 96.0, 192.0])
    )
    tail = replace(
        plateau.tracks[0],
        start_ns=np.full(3, 192.0),
        duration_ns=np.full(3, 2.0),
        amplitude=np.array([0.0, 0.1, 0.2]),
        active=np.array([False, True, True]),
    )
    timeline = replace(
        sampled,
        end_ns=np.array([12.0, 194.0, 194.0]),
        tracks=(
            *(
                replace(t, active=np.array([True, False, False]))
                for t in plateau.tracks
            ),
            *(
                replace(t, active=tail.active, envelopes=(envelope,))
                for t in sampled.tracks
            ),
            tail,
        ),
    )
    capture = CaptureBuffer()
    with closing(model_source()) as source, closing(Engine(1, backend="cpu")) as engine:
        engine.evolve(source, (timeline,), capture=capture)
    metadata = read_capture_metadata(capture.records[0])
    (prefix,) = metadata["prefix_runs"]
    assert {row["point"] for row in prefix["adopters"]} == {1, 2}
    assert any(
        row["point"] == 0 and row["family"] == "multitone_rwa"
        for row in metadata["routes"]
    )
    assert {row["family"] for row in metadata["routes"] if row["prefix_run"] == 0} == {
        "common_frame"
    }


def test_common_frame_selection_and_results_are_row_order_invariant():
    source = model_source()
    with closing(Engine(1)) as engine:
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
    with closing(Engine(1)) as engine:
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
    with closing(Engine(1)) as engine:
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
    with closing(Engine(1)) as engine:
        result = run(engine, source, timeline)
    original = fallback(source, timeline)
    assert result.paths.segments_by_kernel == original.paths.segments_by_kernel
    np.testing.assert_array_equal(result.densities, original.densities)


@pytest.mark.parametrize("workers", [1, 2])
def test_rejected_unique_common_keys_preserve_shared_fallback_preparation(workers):
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
    with closing(Engine(workers, kernels=(common, fallback_kernel))) as engine:
        result = run(engine, source, timeline)
    assert result.paths.segments_by_kernel == {"multitone_rwa": 4}
    assert common.admission_rows == 4
    assert fallback_kernel.preparations == 1
    np.testing.assert_array_equal(
        result.densities, np.repeat(source.initial_density[None], 4, axis=0)
    )
