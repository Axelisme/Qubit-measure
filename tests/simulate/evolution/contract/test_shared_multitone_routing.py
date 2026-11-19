"""Select among the eligible sampled dual-tone routes through Engine inspection."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    Engine,
    EnvelopeSpec,
    SecularDrive,
    SecularModel,
    read_capture_metadata,
)

from .accuracy_cases import AccuracyCase, coherent_source, drive_track, timeline


def _resonant_pair(
    count: int,
    *,
    bath: float = 0.0,
    cutoff: float = 2.7,
    dense: bool = False,
    detuned: bool = False,
) -> AccuracyCase:
    duration = 100.0
    batch = timeline(count, duration)
    knots = np.linspace(0, duration, 15 if dense else 3)
    envelope = EnvelopeSpec(
        "sampled", (1.0 - 0.2 * abs(knots / 50 - 1.0)).astype(complex), knots
    )
    drives = []
    for port, (low, high) in zip(("a", "b"), ((0, 3), (1, 2)), strict=True):
        directed = np.zeros((8, 8), dtype=complex)
        directed[low, high] = 0.002
        if detuned and port == "a":
            directed[0, 1] = 0.0023
        drives.append(
            SecularDrive(port, directed + directed.T, 1j * (directed - directed.T))
        )
    rates = np.zeros((8, 8))
    rates[0, 3] = rates[1, 2] = bath
    model = SecularModel(
        (0,),
        "shared-multitone-routing",
        np.array([0.0, 0.205, 0.505, 0.6, 1.1, 1.3, 1.5, cutoff]),
        np.eye(8, dtype=complex),
        tuple(drives),
        rates,
        np.zeros((8, 8)),
    )
    tracks = tuple(
        replace(
            drive_track(batch, envelope, frequency=frequency, gain=0.08, port=port),
            phase_rad=np.full(count, 0.17),
        )
        for port, frequency in (("a", 0.6), ("b", 0.3))
    )
    return AccuracyCase(
        model.identity, coherent_source(model), replace(batch, tracks=tracks), ()
    )


_CPU_FAMILIES = ("multitone_gate", "multitone_rwa", "dop853")


def _families(
    case: AccuracyCase, families: tuple[str, ...] | None = _CPU_FAMILIES
) -> list[str | None]:
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=families)) as engine,
    ):
        query = engine.inspect_dispatch(case.source, (case.batch,))
    return [row.family for row in query.rows]


def test_cheaper_sampled_rwa_wins_even_when_gate_can_repay_build() -> None:
    assert set(_families(_resonant_pair(64), families=None)) == {"multitone_rwa"}


def test_price_winning_gate_is_used_even_for_a_dissipative_key() -> None:
    case = _resonant_pair(160, bath=0.05)
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=_CPU_FAMILIES)) as engine,
    ):
        result = engine.evolve(case.source, (case.batch,), telemetry=True)
        query = engine.inspect_dispatch(case.source, (case.batch,))

    assert result.paths.segments_by_kernel == {"multitone_gate": 160}
    assert result.paths.routing["multitone_gate"].prepare_key_builds == 1
    assert {row.family for row in query.rows} == {"multitone_gate"}


def test_cheaper_gate_wins_when_both_families_are_admissible() -> None:
    assert set(_families(_resonant_pair(160))) == {"multitone_gate"}


def test_gate_build_failure_aborts_evolve_without_affecting_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zcu_tools.simulate.evolution.kernels import _multitone_gate_phase

    failure = RuntimeError("synthetic gate endpoint failure")

    def fail(*args: object) -> None:
        raise failure

    # The Engine API cannot inject a numerical failure into a valid gate key.
    monkeypatch.setattr(_multitone_gate_phase, "build", fail)
    case = _resonant_pair(160)
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=_CPU_FAMILIES)) as engine,
    ):
        query = engine.inspect_dispatch(case.source, (case.batch,))
        assert {row.family for row in query.rows} == {"multitone_gate"}
        with pytest.raises(
            RuntimeError, match="synthetic gate endpoint failure"
        ) as raised:
            engine.evolve(case.source, (case.batch,))
    assert raised.value is failure


def test_inspection_reports_price_skipped_gate_only_with_allowlist() -> None:
    case = _resonant_pair(1)
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu")) as default,
        closing(Engine(1, backend="cpu", families=_CPU_FAMILIES)) as allowed,
    ):
        (normal,) = default.inspect_dispatch(case.source, (case.batch,)).rows
        (restricted,) = allowed.inspect_dispatch(case.source, (case.batch,)).rows

    assert normal.family == restricted.family == "dop853"
    assert "multitone_gate" not in {candidate.family for candidate in normal.candidates}
    assert {candidate.family: candidate.guard for candidate in restricted.candidates}[
        "multitone_gate"
    ] == "untested_after_short_circuit"


def test_sampled_rwa_refusal_falls_through_to_gate_and_original_by_key() -> None:
    case = _resonant_pair(65, detuned=True)
    tracks = tuple(
        replace(track, amplitude=np.r_[track.amplitude[:-1], 0.081])
        for track in case.batch.tracks
    )
    case = replace(case, batch=replace(case.batch, tracks=tracks))
    assert _families(case) == ["multitone_gate"] * 64 + ["dop853"]


def _phase_groups(case: AccuracyCase, split: int) -> AccuracyCase:
    tracks = tuple(
        replace(
            track,
            phase_rad=np.where(np.arange(len(track.phase_rad)) < split, 0.17, 1.4),
        )
        for track in case.batch.tracks
    )
    return replace(case, batch=replace(case.batch, tracks=tracks))


def test_gate_key_repays_one_build_across_phase_groups_prefixes_and_workers() -> None:
    case = _phase_groups(_resonant_pair(80, cutoff=1.7), split=40)
    batch = case.batch
    # A constant tone after each phase group makes its gate a shared prefix.
    # The separate timeline uses the same gate key but ends without a prefix.
    end = batch.end_ns + 1
    prefix_users = replace(
        batch,
        end_ns=end,
        tracks=(
            *batch.tracks,
            replace(
                drive_track(batch, EnvelopeSpec("constant"), port="a"),
                start_ns=batch.end_ns,
                duration_ns=end - batch.end_ns,
            ),
        ),
    )
    worker = replace(batch.select_points((0,)), points=np.array([80]))
    batches = (prefix_users, worker)
    capture = CaptureBuffer()
    with (
        closing(case.source),
        closing(Engine(2, backend="cpu", families=(*_CPU_FAMILIES, "t6"))) as engine,
    ):
        query = engine.inspect_dispatch(case.source, batches)
        result = engine.evolve(case.source, batches, telemetry=True, capture=capture)
    assert {row.family for row in query.rows if row.key.segment == 0} == {
        "multitone_gate"
    }
    assert result.paths.segments_by_kernel["multitone_gate"] == 81
    assert result.paths.routing["multitone_gate"].prepare_key_builds == 1
    prefixes, local = [read_capture_metadata(record) for record in capture.records]
    assert sorted(len(run["adopters"]) for run in prefixes["prefix_runs"]) == [40, 40]
    (worker_route,) = local["routes"]
    assert worker_route["family"] == "multitone_gate"
    assert worker_route["prefix_run"] is None


def test_gate_rechecks_use_count_after_the_larger_phase_group_selects_rwa() -> None:
    case = _phase_groups(_resonant_pair(80, dense=True), split=79)
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=_CPU_FAMILIES)) as engine,
    ):
        query = engine.inspect_dispatch(case.source, (case.batch,))
        result = engine.evolve(case.source, (case.batch,), telemetry=True)
    assert [row.family for row in query.rows] == ["multitone_rwa"] * 79 + ["dop853"]
    assert result.paths.segments_by_kernel == {"multitone_rwa": 79, "dop853": 1}
    gate = result.paths.routing.get("multitone_gate")
    assert gate is None or gate.prepare_key_builds == 0


def test_two_price_winning_keys_in_one_amplitude_group_each_build_once() -> None:
    count = 160
    case = _resonant_pair(2 * count, bath=0.05)
    tracks = tuple(
        replace(
            track,
            amplitude=np.r_[track.amplitude[:count] * 0.5, track.amplitude[count:]],
        )
        for track in case.batch.tracks
    )
    batch = replace(case.batch, tracks=tracks)
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=_CPU_FAMILIES)) as engine,
    ):
        result = engine.evolve(case.source, (batch,), telemetry=True)
    assert result.paths.segments_by_kernel == {"multitone_gate": 2 * count}
    assert result.paths.routing["multitone_gate"].prepare_key_builds == 2
