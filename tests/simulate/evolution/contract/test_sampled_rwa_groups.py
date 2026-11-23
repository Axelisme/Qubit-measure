"""Engine contracts for the lifetime, identity, and probes of sampled RWA groups."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    Engine,
    EnvelopeSpec,
    ModelSource,
    NoValidKernel,
    SampledRWAProbeMemo,
    TimelineBatch,
    read_capture_metadata,
)

from .accuracy_cases import AccuracyCase, accuracy_case, drive_track, timeline


def test_sampled_multitone_reuses_one_spot_check_only_within_evolve() -> None:
    case = accuracy_case("cheap_sampled_nearby")
    batch = replace(
        case.batch,
        tracks=tuple(
            replace(track, phase_rad=np.repeat(track.phase_rad[:1], 2))
            for track in case.batch.tracks
        ),
    )
    with closing(Engine(1, backend="cpu", families=("multitone_rwa",))) as engine:
        inspection = engine.inspect_dispatch(case.source, (batch,))
        assert len(inspection.rows) == 2
        for row in inspection.rows:
            assert row.family == "multitone_rwa"
            assert any(
                candidate.family == "multitone_rwa" and candidate.guard == "accepted"
                for candidate in row.candidates
            )
        for _ in range(2):
            result = engine.evolve(case.source, (batch,), telemetry=True)
            assert result.paths.segments_by_kernel == {"multitone_rwa": 2}
            assert result.paths.routing["multitone_rwa"].admission_rows == 1


def test_sampled_multitone_rejection_does_not_cross_phase_groups() -> None:
    case = accuracy_case("sampled_multitone_phase_sequence")
    batch = timeline(3, 8.0)
    batch = replace(
        batch,
        tracks=tuple(
            replace(
                drive_track(batch, track.envelopes[0], port=track.port),
                frequency_ghz=np.full(3, track.frequency_ghz[0]),
                phase_rad=np.array([track.phase_rad[0], track.phase_rad[0], -0.9]),
                amplitude=track.amplitude[[1, 0, 0]],
            )
            for track in case.batch.tracks[:2]
        ),
    )
    # Pin the numerical guard so both small groups are compared rather than
    # skipped on price. The passing phase must not appear in the failure.
    capture = CaptureBuffer()
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=("multitone_rwa",))) as engine,
    ):
        with pytest.raises(NoValidKernel) as queried:
            engine.inspect_dispatch(case.source, (batch,))
        with pytest.raises(NoValidKernel) as evolved:
            engine.evolve(case.source, (batch,), capture=capture)
    for result in (queried, evolved):
        assert result.value.attempted == ("multitone_rwa",)
        assert result.value.points == (0, 1)
    metadata = read_capture_metadata(capture.records[0])
    assert not metadata["complete"]
    failed, passed = metadata["sampled_rwa_admissions"]
    assert failed["decision"] == "rejected"
    assert [row["point"] for row in failed["rows"]] == [0, 1]
    (spot,) = failed["spots"]
    assert (spot["batch"], spot["point"], spot["segment"]) == (0, 0, 0)
    assert spot["max_density_error"] is not None and spot["max_density_error"] > 1e-4
    assert passed["decision"] == "accepted"
    assert [row["point"] for row in passed["rows"]] == [2]


def _short_sampled_pair(case: AccuracyCase, count: int) -> TimelineBatch:
    batch = timeline(count, 1.5)
    envelope = EnvelopeSpec(
        "sampled", np.array([0.8, 1.0, 0.9], complex), np.array([0.0, 0.75, 1.5])
    )
    return replace(
        batch,
        tracks=tuple(
            replace(
                drive_track(
                    batch,
                    envelope,
                    frequency=float(track.frequency_ghz[0]),
                    port=track.port,
                ),
                phase_rad=np.full(count, track.phase_rad[0]),
            )
            for track in case.batch.tracks
        ),
    )


@pytest.mark.parametrize(
    "change", ["point", "duration", "rates", "dephasing", "phase", "gain"]
)
def test_sampled_multitone_memo_requires_exact_probe_inputs(change: str) -> None:
    case = accuracy_case("joint_constant")
    batch = _short_sampled_pair(case, 1)
    model = case.source.model((0,))
    changed = replace(batch, points=np.array([13], dtype=np.int64))
    first, second = changed.tracks
    if change == "duration":
        changed = replace(changed, end_ns=changed.end_ns - 0.125)
    elif change == "rates":
        rates = model.rates_per_ns.copy()
        rates[0, 1] += 1e-5
        model = replace(model, rates_per_ns=rates)
    elif change == "dephasing":
        dephasing = model.dephasing_per_ns.copy()
        dephasing[0, 1] += 1e-5
        dephasing[1, 0] += 1e-5
        model = replace(model, dephasing_per_ns=dephasing)
    elif change == "phase":
        changed = replace(
            changed, tracks=(replace(first, phase_rad=first.phase_rad + 0.125), second)
        )
    elif change == "gain":
        changed = replace(
            changed, tracks=(replace(first, amplitude=-first.amplitude), second)
        )

    capture = CaptureBuffer()
    # Keep the model's caller identity even when its loss arrays differ.
    with (
        closing(case.source),
        closing(
            ModelSource(case.source.initial_density, lambda level: model)
        ) as source,
        closing(SampledRWAProbeMemo()) as memo,
        closing(Engine(1, backend="cpu", families=("multitone_rwa",))) as engine,
    ):
        initial = engine.evolve(
            case.source, (batch,), telemetry=True, probe_memo=memo, capture=capture
        )
        result = engine.evolve(
            source, (changed,), telemetry=True, probe_memo=memo, capture=capture
        )
    assert initial.paths.routing["multitone_rwa"].admission_rows == 1
    hit = change == "point"
    routing = result.paths.routing["multitone_rwa"]
    assert routing.admission_rows == int(not hit)
    assert routing.probe_memo_hits == int(hit)
    assert (routing.probe_integration_cpu_s > 0.0) == (not hit)
    if hit:
        assert result.densities.tobytes() == initial.densities.tobytes()
        assert result.paths.segments_by_kernel == initial.paths.segments_by_kernel
        before, after = [read_capture_metadata(record) for record in capture.records]
        (before_group,) = before["sampled_rwa_admissions"]
        (after_group,) = after["sampled_rwa_admissions"]
        assert before_group["decision"] == after_group["decision"] == "accepted"
        assert after_group["spots"] == [{**before_group["spots"][0], "point": 13}]


def test_sampled_multitone_uses_its_own_group_in_shared_prefix() -> None:
    case = accuracy_case("joint_constant")
    scan = _short_sampled_pair(case, 3)
    first, second = scan.tracks
    active = np.array([False, True, True])
    plateau = replace(
        drive_track(scan, EnvelopeSpec("constant"), port=first.port, gain=1e-6),
        start_ns=np.full(3, 1.5),
        phase_rad=np.full(3, 0.4),
    )
    batch = replace(
        scan,
        end_ns=np.full(3, 3.0),
        tracks=(replace(first, active=active), replace(second, active=active), plateau),
    )
    with (
        closing(case.source),
        closing(
            Engine(1, backend="cpu", families=("t4", "t6", "multitone_rwa"))
        ) as engine,
    ):
        result = engine.evolve(case.source, (batch,), telemetry=True)
    assert result.paths.segments_by_kernel["multitone_rwa"] == 2
    assert result.paths.routing["multitone_rwa"].admission_rows == 1


def test_sampled_multitone_probes_each_tones_amplitude_maximum() -> None:
    case = accuracy_case("joint_constant")
    scan = _short_sampled_pair(case, 3)
    batch = replace(
        scan,
        tracks=tuple(
            replace(track, amplitude=np.array(gains))
            for track, gains in zip(
                scan.tracks, ((0.03, 0.015, 0.002), (0.015, 0.031, 0.002)), strict=True
            )
        ),
    )
    first, second = batch.tracks
    changed = replace(
        batch, tracks=(first, replace(second, amplitude=np.array([0.015, 0.03, 0.002])))
    )
    with (
        closing(case.source),
        closing(SampledRWAProbeMemo()) as memo,
        closing(Engine(1, backend="cpu", families=("multitone_rwa",))) as engine,
    ):
        result = engine.evolve(case.source, (batch,), telemetry=True, probe_memo=memo)
        partial = engine.evolve(
            case.source, (changed,), telemetry=True, probe_memo=memo
        )
        cold = engine.evolve(case.source, (changed,))
    assert result.paths.segments_by_kernel == {"multitone_rwa": 3}
    assert result.paths.routing["multitone_rwa"].admission_rows == 2
    routing = partial.paths.routing["multitone_rwa"]
    assert routing.admission_rows == routing.probe_memo_hits == 1
    assert partial.paths.segments_by_kernel == cold.paths.segments_by_kernel
    assert partial.densities.tobytes() == cold.densities.tobytes()


@pytest.mark.parametrize("variation", ["scale", "one_ulp_sample", "sample_grid"])
def test_sampled_multitone_requires_exact_proportional_envelopes(
    variation: str,
) -> None:
    case = accuracy_case("joint_constant")
    scan = _short_sampled_pair(case, 2)
    samples = np.array([0.5 + 0.125j, 1.0, 0.75 - 0.25j], complex)
    scaled = np.asarray(samples * 1.5, dtype=np.complex128)
    if variation == "one_ulp_sample":
        scaled.real[0] = np.nextafter(scaled.real[0], np.inf)
    knots = np.array([0.0, 0.75, 1.5])
    scaled_knots = np.array([0.0, 0.6, 1.5]) if variation == "sample_grid" else knots
    envelopes = (
        EnvelopeSpec("sampled", samples, knots),
        EnvelopeSpec("sampled", scaled, scaled_knots),
    )
    batch = replace(
        scan,
        tracks=tuple(
            replace(
                track,
                amplitude=np.array([0.01, 0.02]),
                envelopes=envelopes,
                envelope_index=np.arange(2),
            )
            for track in scan.tracks
        ),
    )
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=("multitone_rwa",))) as engine,
    ):
        result = engine.evolve(case.source, (batch,), telemetry=True)
    assert result.paths.segments_by_kernel == {"multitone_rwa": 2}
    assert result.paths.routing["multitone_rwa"].admission_rows == (
        1 if variation == "scale" else 2
    )


def test_sampled_multitone_does_not_group_different_pulse_slices() -> None:
    case = accuracy_case("joint_constant")
    scan = _short_sampled_pair(case, 2)
    first, second = scan.tracks
    # Both points resume at 0.5 ns and play a 1 ns pair at identical phases.
    # Only the second point's first tone is already 0.5 ns into its envelope.
    batch = replace(
        scan,
        start_ns=np.full(2, 0.5),
        initial_densities=np.repeat(case.source.initial_density[None], 2, axis=0),
        tracks=(
            replace(
                first, start_ns=np.array([0.5, 0.0]), duration_ns=np.array([1.0, 1.5])
            ),
            replace(second, start_ns=np.full(2, 0.5), duration_ns=np.ones(2)),
        ),
    )
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=("multitone_rwa",))) as engine,
    ):
        result = engine.evolve(case.source, (batch,), telemetry=True)
    assert result.paths.segments_by_kernel == {"multitone_rwa": 2}
    assert result.paths.routing["multitone_rwa"].admission_rows == 2
