"""Read-only constant RWA diagnostics agree with public Engine admission."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate import evolution

from .accuracy_cases import coherent_source, drive_track, linear_drive_model, timeline
from .common_frame_cases import close_carrier_case
from .guard_edge_cases import guard_edge_case
from .multitone_cases import (
    accumulated_pair_case,
    restored_dimension_case,
    short_pair_case,
)


@pytest.mark.sim
@pytest.mark.parametrize(
    ("build_case", "accepted"),
    [
        pytest.param(short_pair_case, True, id="short-pair-accepted"),
        pytest.param(accumulated_pair_case, False, id="a16-secular-rejected"),
    ],
)
def test_multitone_diagnostic_matches_admission(
    build_case: Callable[[], tuple[evolution.ModelSource, evolution.TimelineBatch]],
    accepted: bool,
) -> None:
    source, batch = build_case()
    with (
        closing(source),
        closing(evolution.Engine(1, families=("multitone_rwa",))) as engine,
    ):
        pump, probe = batch.tracks
        level = tuple(int(value) for value in batch.flux.level[0])
        frequencies = (float(pump.frequency_ghz[0]), float(probe.frequency_ghz[0]))
        budget = 1e-3
        diagnostic = evolution.multitone_rwa_diagnostic(
            source.model(level),
            (pump.port, probe.port),
            frequencies,
            (float(pump.amplitude[0]), float(probe.amplitude[0])),
            float(batch.end_ns[0]),
            budget=budget,
        )
        metrics = tuple(
            expect_metric(metric)
            for metric in (
                diagnostic.budget,
                diagnostic.carrier_ghz,
                diagnostic.omission_transfer,
                diagnostic.omission_phase_rad,
                diagnostic.real_dimension,
                diagnostic.secular_accumulation,
            )
        )
        allowance, carrier, transfer, phase, dimension, secular = metrics
        assert all(np.isfinite(metric.value) for metric in metrics)
        assert all(np.isfinite(metric.threshold) for metric in metrics)
        assert all(metric.value >= 0 for metric in metrics)
        assert tuple(metric.comparison for metric in metrics) == (
            ">=",
            ">=",
            "<=",
            "<=",
            "<=",
            "<=",
        )
        assert allowance.value == budget
        assert carrier.value == min(abs(frequency) for frequency in frequencies)
        assert all(metric.passed for metric in metrics[:-1])
        assert secular.passed is accepted
        assert diagnostic.accepted is accepted
        assert diagnostic.accepted == all(metric.passed for metric in metrics)
        if accepted:
            assert transfer.value == 0.0
            assert phase.value == 0.0
            assert dimension.value == 4
            assert secular.value == 0.0
            result = engine.evolve(source, (batch,), budget=budget, telemetry=True)
            assert result.paths.segments_by_kernel == {"multitone_rwa": 1}
        else:
            assert secular.value > secular.threshold
            with pytest.raises(evolution.NoValidKernel):
                engine.evolve(source, (batch,), budget=budget)


@pytest.mark.sim
def test_multitone_same_port_is_rejected_query() -> None:
    source, batch = short_pair_case()
    with (
        closing(source),
        closing(evolution.Engine(1, families=("multitone_rwa",))) as engine,
    ):
        batch = replace(
            batch, tracks=tuple(replace(track, port="a") for track in batch.tracks)
        )
        pump, probe = batch.tracks
        level = tuple(int(value) for value in batch.flux.level[0])
        budget = 1e-3
        diagnostic = evolution.multitone_rwa_diagnostic(
            source.model(level),
            (pump.port, probe.port),
            (float(pump.frequency_ghz[0]), float(probe.frequency_ghz[0])),
            (float(pump.amplitude[0]), float(probe.amplitude[0])),
            float(batch.end_ns[0]),
            budget=budget,
        )
        assert diagnostic.accepted is False
        with pytest.raises(evolution.NoValidKernel):
            engine.evolve(source, (batch,), budget=budget)


@pytest.mark.sim
@pytest.mark.parametrize(
    ("ports", "frequencies", "amplitudes", "duration", "budget"),
    [
        pytest.param(
            ("a", "b"),
            (float("nan"), 0.37),
            (1.0, 1.0),
            8.0,
            1e-3,
            id="first-frequency-nan",
        ),
        pytest.param(
            ("a", "b"),
            (float("inf"), 0.37),
            (1.0, 1.0),
            8.0,
            1e-3,
            id="first-frequency-positive-inf",
        ),
        pytest.param(
            ("a", "b"),
            (-float("inf"), 0.37),
            (1.0, 1.0),
            8.0,
            1e-3,
            id="first-frequency-negative-inf",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, float("nan")),
            (1.0, 1.0),
            8.0,
            1e-3,
            id="second-frequency-nan",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, float("inf")),
            (1.0, 1.0),
            8.0,
            1e-3,
            id="second-frequency-positive-inf",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, -float("inf")),
            (1.0, 1.0),
            8.0,
            1e-3,
            id="second-frequency-negative-inf",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (float("nan"), 1.0),
            8.0,
            1e-3,
            id="first-amplitude-nan",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (float("inf"), 1.0),
            8.0,
            1e-3,
            id="first-amplitude-positive-inf",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (-float("inf"), 1.0),
            8.0,
            1e-3,
            id="first-amplitude-negative-inf",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, float("nan")),
            8.0,
            1e-3,
            id="second-amplitude-nan",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, float("inf")),
            8.0,
            1e-3,
            id="second-amplitude-positive-inf",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, -float("inf")),
            8.0,
            1e-3,
            id="second-amplitude-negative-inf",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, 1.0),
            float("nan"),
            1e-3,
            id="duration-nan",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, 1.0),
            float("inf"),
            1e-3,
            id="duration-positive-inf",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, 1.0),
            -float("inf"),
            1e-3,
            id="duration-negative-inf",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, 1.0),
            -1.0,
            1e-3,
            id="duration-negative-one-ns",
        ),
        pytest.param(
            ("unknown", "b"),
            (0.3, 0.37),
            (1.0, 1.0),
            8.0,
            1e-3,
            id="first-port-unknown",
        ),
        pytest.param(
            ("a", "unknown"),
            (0.3, 0.37),
            (1.0, 1.0),
            8.0,
            1e-3,
            id="second-port-unknown",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, 1.0),
            8.0,
            float("nan"),
            id="budget-nan",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, 1.0),
            8.0,
            float("inf"),
            id="budget-positive-inf",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, 1.0),
            8.0,
            -float("inf"),
            id="budget-negative-inf",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, 1.0),
            8.0,
            0.0,
            id="budget-zero",
        ),
        pytest.param(
            ("a", "b"),
            (0.3, 0.37),
            (1.0, 1.0),
            8.0,
            -1.0,
            id="budget-negative-one",
        ),
    ],
)
def test_multitone_invalid_row_raises(
    ports: tuple[str, str],
    frequencies: tuple[float, float],
    amplitudes: tuple[float, float],
    duration: float,
    budget: float,
) -> None:
    source, _ = short_pair_case()
    with (
        closing(source),
        pytest.raises(
            ValueError,
            match=(
                "constant RWA requires finite carrier/gain and nonnegative finite duration"
                "|unknown constant RWA drive port: unknown"
                "|budget must be positive and finite"
            ),
        ),
    ):
        evolution.multitone_rwa_diagnostic(
            source.model((0,)), ports, frequencies, amplitudes, duration, budget=budget
        )


@pytest.mark.sim
@pytest.mark.parametrize(
    ("budget", "first_frequency", "budget_passed"),
    [
        pytest.param(5e-4, 0.6, False, id="low-budget"),
        pytest.param(1e-3, 0.04, True, id="low-positive-carrier"),
        pytest.param(1e-3, -0.04, True, id="low-negative-carrier"),
        pytest.param(1e-3, 0.0, True, id="zero-carrier"),
    ],
)
def test_multitone_early_rejection_preserves_uncomputed(
    budget: float, first_frequency: float, budget_passed: bool
) -> None:
    source, batch = short_pair_case()
    with (
        closing(source),
        closing(evolution.Engine(1, families=("multitone_rwa",))) as engine,
    ):
        pump, probe = batch.tracks
        batch = replace(
            batch,
            tracks=(
                replace(pump, frequency_ghz=np.array([first_frequency])),
                probe,
            ),
        )
        pump, probe = batch.tracks
        level = tuple(int(value) for value in batch.flux.level[0])
        frequencies = (float(pump.frequency_ghz[0]), float(probe.frequency_ghz[0]))
        diagnostic = evolution.multitone_rwa_diagnostic(
            source.model(level),
            (pump.port, probe.port),
            frequencies,
            (float(pump.amplitude[0]), float(probe.amplitude[0])),
            float(batch.end_ns[0]),
            budget=budget,
        )
        assert diagnostic.accepted is False
        allowance = expect_metric(diagnostic.budget)
        assert allowance.value == budget
        assert allowance.passed is budget_passed
        if budget_passed:
            carrier = expect_metric(diagnostic.carrier_ghz)
            assert carrier.value == min(abs(frequency) for frequency in frequencies)
            assert carrier.passed is False
        else:
            assert isinstance(diagnostic.carrier_ghz, evolution.RWAUncomputed)
        for metric in (
            diagnostic.omission_transfer,
            diagnostic.omission_phase_rad,
            diagnostic.real_dimension,
            diagnostic.secular_accumulation,
        ):
            assert isinstance(metric, evolution.RWAUncomputed)
        with pytest.raises(evolution.NoValidKernel):
            engine.evolve(source, (batch,), budget=budget)


@pytest.mark.sim
def test_common_frame_diagnostic_matches_accepted_row() -> None:
    model = linear_drive_model()
    batch = common_frame_batch((0.6, 0.63), (0.08, 0.08), 8.0)
    assert hasattr(evolution, "common_frame_diagnostic"), (
        "common-frame guard must expose its computed row-local indicators"
    )
    diagnostic = evolution.common_frame_diagnostic(
        model, ("drive", "drive"), (0.6, 0.63), (0.08, 0.08), 8.0
    )
    assert diagnostic.accepted
    assert diagnostic.reason == "accepted"
    assert expect_metric(diagnostic.coherent).passed
    assert expect_metric(diagnostic.bath).passed
    assert 0 < expect_metric(diagnostic.coherent).value < 1e-3
    assert 0 < expect_metric(diagnostic.bath).value < 1e-3
    with (
        closing(coherent_source(model)) as source,
        closing(evolution.Engine(1, families=("common_frame",))) as engine,
    ):
        result = engine.evolve(source, (batch,), telemetry=True)
        assert result.paths.segments_by_kernel == {"common_frame": 1}
        assert result.paths.routing["common_frame"].admission_rows == 1
        assert result.paths.routing["common_frame"].rejected_rows == 0


def common_frame_batch(
    frequencies: tuple[float, float],
    amplitudes: tuple[float, float],
    duration: float,
    envelopes: tuple[evolution.EnvelopeSpec, evolution.EnvelopeSpec] | None = None,
) -> evolution.TimelineBatch:
    """Build a full-support synthetic two-tone row on the drive port."""
    batch = timeline(1, duration)
    envelopes = envelopes or (
        evolution.EnvelopeSpec("constant"),
        evolution.EnvelopeSpec("constant"),
    )
    return replace(
        batch,
        tracks=tuple(
            drive_track(batch, envelope, frequency=f, gain=g)
            for f, g, envelope in zip(frequencies, amplitudes, envelopes, strict=True)
        ),
    )


@pytest.mark.sim
@pytest.mark.parametrize("spacing", [0.0005, 0.002])
def test_common_frame_six_level_close_carriers_match_guard(spacing: float) -> None:
    # F K6-type input: six levels, near-degenerate carriers and a weaker probe.
    x = np.diag(np.full(5, 0.002), 1).astype(np.complex128)
    x += x.conj().T
    model = evolution.SecularModel(
        (0,),
        "diagnostic-six-level",
        np.array([0.0, 0.6, 1.23, 1.89, 2.58, 3.3]),
        np.eye(6, dtype=np.complex128),
        (evolution.SecularDrive("drive", x, np.zeros_like(x)),),
        np.zeros((6, 6)),
        np.zeros((6, 6)),
    )
    frequencies = (0.6, 0.6 + spacing)
    diagnostic = evolution.common_frame_diagnostic(
        model, ("drive", "drive"), frequencies, (0.05, 0.005), 24.0
    )
    assert diagnostic.accepted
    assert expect_metric(diagnostic.coherent).value > 0
    assert expect_metric(diagnostic.bath).value == 0
    assert diagnostic.derivative == 0
    with (
        closing(coherent_source(model)) as source,
        closing(evolution.Engine(1, families=("common_frame",))) as engine,
    ):
        result = engine.evolve(
            source,
            (common_frame_batch(frequencies, (0.05, 0.005), 24.0),),
            telemetry=True,
        )
        assert result.paths.segments_by_kernel == {"common_frame": 1}
        assert result.paths.routing["common_frame"].rejected_rows == 0


@pytest.mark.sim
@pytest.mark.parametrize(
    "frequencies,reason",
    [((0.6, -0.6), "carrier"), ((0.6, 2.4), "denominator")],
)
def test_common_frame_early_rejection_leaves_indicators_uncomputed(
    frequencies: tuple[float, float], reason: str
) -> None:
    model = linear_drive_model()
    diagnostic = evolution.common_frame_diagnostic(
        model, ("drive", "drive"), frequencies, (0.08, 0.08), 8.0
    )
    assert not diagnostic.accepted
    assert diagnostic.reason == reason
    assert diagnostic.coherent == evolution.RWAUncomputed()
    assert diagnostic.bath == evolution.RWAUncomputed()
    assert diagnostic.eta == evolution.RWAUncomputed()
    assert diagnostic.second == evolution.RWAUncomputed()
    assert diagnostic.third == evolution.RWAUncomputed()
    assert diagnostic.fourth == evolution.RWAUncomputed()
    assert diagnostic.tail == evolution.RWAUncomputed()
    assert diagnostic.derivative == evolution.RWAUncomputed()
    if reason == "carrier":
        assert diagnostic.denominator_min_ghz == evolution.RWAUncomputed()
    else:
        assert diagnostic.denominator_min_ghz == pytest.approx(0.0)
    with (
        closing(coherent_source(model)) as source,
        closing(evolution.Engine(1, families=("common_frame",))) as engine,
        pytest.raises(evolution.NoValidKernel),
    ):
        engine.evolve(source, (common_frame_batch(frequencies, (0.08, 0.08), 8.0),))


@pytest.mark.sim
@pytest.mark.parametrize("reason", ["coherent", "bath", "derivative"])
def test_common_frame_numeric_rejection_keeps_computed_metrics(reason: str) -> None:
    model = linear_drive_model()
    gains = (50.0, 50.0) if reason == "coherent" else (0.08, 0.08)
    if reason == "bath":
        model = replace(model, dephasing_per_ns=(1 - np.eye(2)) * 1e6)
    envelope = (
        evolution.EnvelopeSpec(
            "sampled", np.array([1, -1], dtype=np.complex128), np.array([0.0, 1e-6])
        )
        if reason == "derivative"
        else evolution.EnvelopeSpec("constant")
    )
    diagnostic = evolution.common_frame_diagnostic(
        model,
        ("drive", "drive"),
        (0.6, 0.63),
        gains,
        8.0,
        envelopes=(envelope, envelope),
    )
    assert not diagnostic.accepted
    assert diagnostic.reason == ("coherent" if reason == "derivative" else reason)
    coherent = expect_metric(diagnostic.coherent)
    bath = expect_metric(diagnostic.bath)
    assert coherent.passed is (reason == "bath")
    assert bath.passed is (reason != "bath")
    assert isinstance(diagnostic.derivative, float)
    assert (diagnostic.derivative > 0) is (reason == "derivative")
    with (
        closing(coherent_source(model)) as source,
        closing(evolution.Engine(1, families=("common_frame",))) as engine,
        pytest.raises(evolution.NoValidKernel),
    ):
        engine.evolve(
            source, (common_frame_batch((0.6, 0.63), gains, 8.0, (envelope, envelope)),)
        )


@pytest.mark.sim
def test_common_frame_zero_gain_reports_zero_computed_indicators() -> None:
    diagnostic = evolution.common_frame_diagnostic(
        linear_drive_model(), ("drive", "drive"), (0.6, 0.63), (0.0, 0.0), 8.0
    )
    assert diagnostic.accepted
    assert expect_metric(diagnostic.coherent).value == 0
    assert expect_metric(diagnostic.bath).value == 0
    assert diagnostic.eta == diagnostic.second == diagnostic.third == 0
    assert diagnostic.fourth == diagnostic.tail == diagnostic.derivative == 0


@pytest.mark.sim
@pytest.mark.parametrize(
    "frequencies,amplitudes,duration,ports",
    [
        ((np.nan, 0.63), (0.08, 0.08), 8.0, ("drive", "drive")),
        ((0.6, 0.63), (np.inf, 0.08), 8.0, ("drive", "drive")),
        ((0.6, 0.63), (0.08, 0.08), -1.0, ("drive", "drive")),
        ((0.6, 0.63), (0.08, 0.08), np.inf, ("drive", "drive")),
        ((0.6, 0.63), (0.08, 0.08), 8.0, ("unknown", "drive")),
    ],
)
def test_common_frame_invalid_row_raises(
    frequencies: tuple[float, float],
    amplitudes: tuple[float, float],
    duration: float,
    ports: tuple[str, str],
) -> None:
    with pytest.raises(ValueError, match="finite|nonnegative|unknown"):
        evolution.common_frame_diagnostic(
            linear_drive_model(), ports, frequencies, amplitudes, duration
        )


def expect_metric(
    metric: evolution.RWAGuardMetric | evolution.RWAUncomputed,
) -> evolution.RWAGuardMetric:
    """Require an actually computed numeric value, not a skipped metric."""
    assert isinstance(metric, evolution.RWAGuardMetric)
    return metric


@pytest.mark.sim
@pytest.mark.parametrize(
    "scale,expected_phase,accepted",
    [(1.0, 0.05999994, True), (np.sqrt(1.000001 / 0.999999), 0.06000006, False)],
)
def test_phase_edge_diagnostic_matches_routing(
    scale: float, expected_phase: float, accepted: bool
) -> None:
    case = guard_edge_case("single_tone_rwa")
    track = case.batch.tracks[0]
    amplitude = float(track.amplitude[0]) * scale
    batch = replace(
        case.batch, tracks=(replace(track, amplitude=track.amplitude * scale),)
    )
    assert hasattr(evolution, "single_tone_rwa_diagnostic"), (
        "constant RWA guard must expose its numeric admission diagnostics"
    )
    diagnostic = evolution.single_tone_rwa_diagnostic(
        linear_drive_model(), "drive", 0.6, amplitude, 8.0
    )
    assert expect_metric(diagnostic.phase_rad).value == pytest.approx(expected_phase)
    assert expect_metric(diagnostic.phase_rad).passed is accepted
    assert diagnostic.accepted is accepted
    assert expect_metric(diagnostic.omission_transfer).passed
    assert expect_metric(diagnostic.strength_ratio).passed
    if accepted:
        assert expect_metric(diagnostic.pruning).passed
        assert expect_metric(diagnostic.real_dimension).passed
    else:
        assert diagnostic.pruning == evolution.RWAUncomputed()
        assert diagnostic.real_dimension == evolution.RWAUncomputed()
    with closing(case.source), closing(evolution.Engine(1)) as engine:
        result = engine.evolve(case.source, (batch,))
        assert dict(result.paths.segments_by_kernel) == {
            "single_tone_rwa" if accepted else "floquet_period": 2
        }
        # Reading diagnostics cannot alter the source's later pinned admission.
        with closing(evolution.Engine(1, families=("single_tone_rwa", "t4"))) as pinned:
            if accepted:
                result = pinned.evolve(case.source, (batch,))
                assert dict(result.paths.segments_by_kernel) == {"single_tone_rwa": 2}
            else:
                with pytest.raises(RuntimeError, match="domain plateau"):
                    pinned.evolve(case.source, (batch,))


@pytest.mark.sim
@pytest.mark.parametrize("frequency", [0.049999, -0.049999, 0.0])
def test_low_carrier_reports_rejection_without_later_metrics(frequency: float) -> None:
    model = linear_drive_model()
    diagnostic = evolution.single_tone_rwa_diagnostic(
        model, "drive", frequency, 0.08, 8.0
    )
    assert not diagnostic.accepted
    assert expect_metric(diagnostic.carrier_ghz).value == abs(frequency)
    assert not expect_metric(diagnostic.carrier_ghz).passed
    assert diagnostic.omission_transfer == evolution.RWAUncomputed()
    assert diagnostic.strength_ratio == evolution.RWAUncomputed()
    assert diagnostic.phase_rad == evolution.RWAUncomputed()
    assert diagnostic.pruning == evolution.RWAUncomputed()
    assert diagnostic.real_dimension == evolution.RWAUncomputed()
    batch = timeline(1, 8.0)
    batch = replace(
        batch,
        tracks=(
            drive_track(
                batch,
                evolution.EnvelopeSpec("constant"),
                frequency=frequency,
                gain=0.08,
            ),
        ),
    )
    with (
        closing(coherent_source(model)) as source,
        closing(evolution.Engine(1, families=("single_tone_rwa", "t4"))) as engine,
        pytest.raises(RuntimeError, match="domain (slow|plateau)"),
    ):
        engine.evolve(source, (batch,))


@pytest.mark.sim
def test_common_frame_rejects_close_six_level_carriers() -> None:
    case = close_carrier_case()
    model = case.source.model((0,))
    with (
        closing(case.source),
        closing(evolution.Engine(1, families=("common_frame",))) as engine,
        pytest.raises(evolution.NoValidKernel),
    ):
        engine.evolve(case.source, (case.batch,))
    pump, probe = case.batch.tracks
    diagnostic = evolution.common_frame_diagnostic(
        model,
        (pump.port, probe.port),
        (float(pump.frequency_ghz[0]), float(probe.frequency_ghz[0])),
        (float(pump.amplitude[0]), float(probe.amplitude[0])),
        float(case.batch.end_ns[0]),
    )
    assert not diagnostic.accepted
    assert diagnostic.reason == "coherent"
    assert not expect_metric(diagnostic.coherent).passed
    assert expect_metric(diagnostic.bath).passed


@pytest.mark.sim
def test_multitone_restored_dimension_hides_stale_metrics() -> None:
    source, batch = restored_dimension_case()
    with (
        closing(source),
        closing(evolution.Engine(1, families=("multitone_rwa",))) as engine,
    ):
        pump, probe = batch.tracks
        level = tuple(int(value) for value in batch.flux.level[0])
        frequencies = (float(pump.frequency_ghz[0]), float(probe.frequency_ghz[0]))
        budget = 1e-3
        diagnostic = evolution.multitone_rwa_diagnostic(
            source.model(level),
            (pump.port, probe.port),
            frequencies,
            (float(pump.amplitude[0]), float(probe.amplitude[0])),
            float(batch.end_ns[0]),
            budget=budget,
        )
        allowance = expect_metric(diagnostic.budget)
        assert allowance.value == budget
        assert allowance.passed is True
        carrier = expect_metric(diagnostic.carrier_ghz)
        assert carrier.value == min(abs(frequency) for frequency in frequencies)
        assert carrier.passed is True
        dimension = expect_metric(diagnostic.real_dimension)
        assert dimension.value == 289
        assert dimension.passed is False
        assert diagnostic.accepted is False
        for metric in (
            diagnostic.omission_transfer,
            diagnostic.omission_phase_rad,
            diagnostic.secular_accumulation,
        ):
            assert isinstance(metric, evolution.RWAUncomputed)
        with pytest.raises(evolution.NoValidKernel):
            engine.evolve(source, (batch,), budget=budget)
