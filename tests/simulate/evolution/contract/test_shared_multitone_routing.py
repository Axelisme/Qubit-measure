"""Select among the eligible sampled dual-tone routes through Engine inspection."""

from contextlib import closing
from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import (
    Engine,
    EnvelopeSpec,
    SecularDrive,
    SecularModel,
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


def test_gate_split_refusal_falls_through_to_sampled_rwa() -> None:
    assert set(_families(_resonant_pair(160, bath=0.05))) == {"multitone_rwa"}


def test_cheaper_gate_wins_when_both_families_are_admissible() -> None:
    assert set(_families(_resonant_pair(160))) == {"multitone_gate"}


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


def test_gate_key_repays_one_build_across_different_phase_groups() -> None:
    case = _phase_groups(_resonant_pair(80, cutoff=1.7), split=40)
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=_CPU_FAMILIES)) as engine,
    ):
        query = engine.inspect_dispatch(case.source, (case.batch,))
        result = engine.evolve(case.source, (case.batch,), telemetry=True)
    assert {row.family for row in query.rows} == {"multitone_gate"}
    assert result.paths.segments_by_kernel == {"multitone_gate": 80}
    assert result.paths.routing["multitone_gate"].prepare_key_builds == 1


def test_gate_rechecks_use_count_after_the_larger_phase_group_selects_rwa() -> None:
    case = _phase_groups(_resonant_pair(100, dense=True), split=99)
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=_CPU_FAMILIES)) as engine,
    ):
        query = engine.inspect_dispatch(case.source, (case.batch,))
        result = engine.evolve(case.source, (case.batch,), telemetry=True)
    assert [row.family for row in query.rows] == ["multitone_rwa"] * 99 + ["dop853"]
    assert result.paths.segments_by_kernel == {"multitone_rwa": 99, "dop853": 1}
    assert "multitone_gate" not in result.paths.routing


def test_a_rejected_gate_key_does_not_displace_a_successful_key_in_its_group() -> None:
    case = _resonant_pair(320, bath=0.05)
    tracks = tuple(
        replace(track, amplitude=np.r_[np.full(160, 0.0001), track.amplitude[160:]])
        for track in case.batch.tracks
    )
    case = replace(case, batch=replace(case.batch, tracks=tracks))
    assert _families(case) == ["multitone_gate"] * 160 + ["dop853"] * 160
