"""Engine routing boundaries separate dispatch costs from numerical admission.

Sampled multitone rejection applies on both sides of the 19/20 cost split.
Pinned calls cannot restore admission; no numerical fix is assumed to reopen it.
"""

import sys
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, NoValidKernel

from .accuracy_cases import (
    AccuracyCase,
    accuracy_case,
    coherent_source,
    drive_track,
    linear_drive_model,
    timeline,
)
from .reference import converged_reference, density_error


def test_explicit_opencl_raises_during_construction_without_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Replace only the optional loader's discovery boundary; no driver needed.
    try:
        import pyopencl as cl
    except ImportError:
        monkeypatch.setitem(
            sys.modules, "pyopencl", SimpleNamespace(get_platforms=list)
        )
    else:
        monkeypatch.setattr(cl, "get_platforms", list)
    with pytest.raises(RuntimeError, match="cl_khr_fp64"):
        Engine(1, backend="opencl")


@pytest.mark.requires_opencl
def test_opencl_keeps_clipped_pulses_and_touching_dop853_chains_on_cpu() -> None:
    case = accuracy_case("sampled_gaussian_9")
    full = case.batch
    clipped = replace(
        full,
        points=full.points + 2,
        tracks=tuple(replace(t, duration_ns=t.duration_ns + 10) for t in full.tracks),
    )
    split = replace(full, points=full.points + 4, segment_boundaries=((50.0,), (50.0,)))
    batches = (full, clipped, split)
    with (
        closing(Engine(2, backend="cpu", families=case.families)) as cpu,
        closing(Engine(2, backend="opencl", families=case.families)) as gpu,
    ):
        expected = cpu.evolve(case.source, batches)
        actual = gpu.evolve(case.source, batches, telemetry=True)
    assert actual.paths.segments_by_kernel == {"dop853": 8}
    assert actual.paths.routing["dop853"].opencl_rows == len(full.points)
    np.testing.assert_array_equal(actual.points, expected.points)
    assert actual.densities[2:].tobytes() == expected.densities[2:].tobytes()
    assert density_error(actual.densities[:2], expected.densities[:2]) < 1e-3


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


@pytest.mark.parametrize("count", [19, 20])
def test_whitelisted_multitone_gate_cost_boundary(count: int) -> None:
    """Neither side of the cost boundary can revive sampled admission."""
    case = _joint_gate_case(count)
    with (
        closing(Engine(1, families=("multitone_rwa", "multitone_gate"))) as engine,
        pytest.raises(NoValidKernel) as rejected,
    ):
        engine.evolve(case.source, (case.batch,))
    assert rejected.value.domain == "multiple"
    assert rejected.value.points == tuple(range(count))
    assert set(rejected.value.attempted) == {"multitone_rwa", "multitone_gate"}

    reference = converged_reference(case.source, case.batch)
    with closing(Engine(1)) as engine:
        for _ in range(2):
            result = engine.evolve(case.source, (case.batch,))
            assert result.paths.segments_by_kernel == {"common_frame": count}
            assert density_error(result.densities, reference.density) <= 2e-3


def test_pinned_multitone_gate_below_cost_break_even() -> None:
    """Pinning bypasses price, not sampled rejection, even after warm fallback."""
    case = _joint_gate_case(19)
    with closing(Engine(1)) as engine:
        engine.evolve(case.source, (case.batch,))
    for family in ("multitone_gate", "multitone_rwa"):
        with (
            closing(Engine(1, families=(family,))) as engine,
            pytest.raises(NoValidKernel) as rejected,
        ):
            engine.evolve(case.source, (case.batch,))
        assert rejected.value.domain == "multiple"
        assert rejected.value.attempted == (family,)


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
