"""Engine routing boundaries separate dispatch costs from numerical admission."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec

from .accuracy import require_families
from .accuracy_cases import (
    AccuracyCase,
    accuracy_case,
    coherent_source,
    drive_track,
    linear_drive_model,
    timeline,
)
from .reference import converged_reference, density_error


def _joint_gate_case(count: int) -> AccuracyCase:
    case = accuracy_case("joint_gate_inventory")
    batch = timeline(count, 1.5)
    tracks = tuple(
        drive_track(
            batch,
            track.envelopes[0],
            frequency=float(track.frequency_ghz[0]),
            gain=float(track.amplitude[0]),
            port=track.port,
        )
        for track in case.batch.tracks
    )
    return replace(case, batch=replace(batch, tracks=tracks))


@pytest.mark.parametrize(
    "count,family", [(19, "multitone_rwa"), (20, "multitone_gate")]
)
def test_whitelisted_multitone_gate_cost_boundary(count: int, family: str) -> None:
    """Keep the whitelisted 19/20 split and unrestricted default density accuracy."""
    case = _joint_gate_case(count)
    reference = converged_reference(case.source, case.batch)
    with closing(Engine(1, families=("multitone_rwa", "multitone_gate"))) as engine:
        result = engine.evolve(case.source, (case.batch,))
    assert dict(result.paths.segments_by_kernel) == {family: count}
    assert density_error(result.densities, reference.density) <= 2e-3

    with closing(Engine(1)) as engine:
        default_result = engine.evolve(case.source, (case.batch,))
    assert density_error(default_result.densities, reference.density) <= 2e-3


def test_pinned_multitone_gate_below_cost_break_even() -> None:
    case = _joint_gate_case(19)
    reference = converged_reference(case.source, case.batch)
    with closing(Engine(1, families=("multitone_gate",))) as engine:
        result = engine.evolve(case.source, (case.batch,))
    require_families(result.paths, ("multitone_gate",))
    assert density_error(result.densities, reference.density) <= 2e-3

    # Pinning bypasses cost, not the shared RWA minimum-budget admission.
    with (
        closing(Engine(1, families=("multitone_gate",))) as engine,
        pytest.raises(RuntimeError, match=r"domain multiple: points \[0, 1,"),
    ):
        engine.evolve(case.source, (case.batch,), budget=1e-4)


def _carrier_case(frequency: float) -> AccuracyCase:
    # Near-resonant strength excludes weak-drive Dyson above the boundary.
    model = replace(linear_drive_model(), energies_ghz=np.array([0.0, 0.05]))
    batch = timeline(2, 1.0)
    batch = replace(
        batch,
        tracks=(
            drive_track(batch, EnvelopeSpec("constant"), frequency=frequency, gain=20),
        ),
    )
    return AccuracyCase("carrier_boundary", coherent_source(model), batch, ())


@pytest.mark.parametrize(
    "frequency,family",
    [
        (sign * carrier, family)
        for sign in (1, -1)
        for carrier, family in (
            (0.029999, "slow_envelope"),
            (0.03, "floquet_period"),
            (0.030001, "floquet_period"),
        )
    ],
)
def test_default_absolute_carrier_boundary(frequency: float, family: str) -> None:
    case = _carrier_case(frequency)
    reference = converged_reference(case.source, case.batch)
    with closing(Engine(1)) as engine:
        result = engine.evolve(case.source, (case.batch,))
    assert dict(result.paths.segments_by_kernel) == {family: 2}
    assert density_error(result.densities, reference.density) <= 2e-3


def test_pinned_slow_envelope_rejects_exact_carrier_boundary() -> None:
    for frequency in (0.03, -0.03):
        case = _carrier_case(frequency)
        with (
            closing(Engine(1, families=("slow_envelope",))) as engine,
            pytest.raises(RuntimeError, match=r"domain plateau: points \[0, 1\]"),
        ):
            engine.evolve(case.source, (case.batch,))
