"""Shared no-drive prefixes and actual-input Dyson at the public engine seam."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    FluxWaveform,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.kernel import SegmentTable
from zcu_tools.simulate.evolution.kernels.dyson import DysonKernel
from zcu_tools.simulate.evolution.kernels.floquet_period import FloquetPeriodKernel
from zcu_tools.simulate.evolution.kernels.t4 import T4


class Fallback(T4):
    """Cheap identity fallback isolates Dyson dispatch from other algorithms."""

    name = "rwa"

    def __init__(self, name="rwa"):
        self.name = name

    def guard(self, table, rows, source, budget):
        return np.ones(len(rows), dtype=bool)

    def apply(self, prepared, key_index, table, rows, states, budget):
        return np.zeros(len(rows), dtype=bool)


def source(energies=(0.0, 0.1, 0.3)):
    x = np.array([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]], complex)
    y = np.zeros((3, 3), complex)
    angle = 0.47
    basis = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        complex,
    )

    def build(level):
        return SecularModel(
            level,
            str((energies, level)),
            np.array(energies),
            np.eye(3, dtype=complex) if level == (0,) else basis,
            (SecularDrive("drive", x, y),),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
        )

    return ModelSource(np.diag([1.0, 0.0, 0.0]).astype(complex), build)


def batch(starts=(7.0, 7.0), *, gain=0.002, frequency=0.4, duration=40.0):
    starts = np.asarray(starts)
    n = len(starts)
    track = DriveTrack(
        "drive",
        starts,
        np.full(n, duration),
        np.full(n, frequency),
        np.zeros(n),
        np.full(n, gain),
        (EnvelopeSpec("constant"),),
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=bool),
    )
    return TimelineBatch(
        np.arange(n),
        (track,),
        FluxSchedule(
            np.arange(n + 1) * 2, np.tile([0.0, 2.0], n), np.tile([[1], [0]], (n, 1))
        ),
        starts + duration,
    )


def evolve(src, b):
    with closing(
        Engine(
            1,
            100000,
            kernels=(
                T4(),
                DysonKernel(),
                Fallback(),
                FloquetPeriodKernel(),
                Fallback("t6"),
            ),
        )
    ) as engine:
        return engine.evolve(src, (b,), cancel=Event())


def test_only_repeated_no_drive_prefixes_bypass_population_bound():
    src, b = source(), batch()
    result = evolve(src, b)
    assert result.paths.segments_by_kernel == {"t4": 4, "dyson": 2}

    # Unequal idle tails, earlier drive, and zero-start inputs retain their
    # ordinary guard. A duplicate final level alone is not a shared prefix.
    unequal = evolve(src, batch((7.0, 8.0)))
    assert unequal.paths.segments_by_kernel == {"t4": 4, "rwa": 2}
    earlier = replace(b.tracks[0], start_ns=np.zeros(2), duration_ns=np.ones(2))
    driven = evolve(src, replace(b, tracks=(earlier, *b.tracks)))
    # The two start-zero prefix pulses retain Dyson; the later probes use T6.
    assert driven.paths.segments_by_kernel == {"dyson": 2, "t4": 4, "t6": 2}
    zero = replace(
        b,
        tracks=(replace(b.tracks[0], start_ns=np.zeros(2)),),
        flux=FluxSchedule(np.arange(3), np.zeros(2), np.zeros((2, 1), dtype=np.int64)),
        end_ns=np.full(2, 40.0),
    )
    # Diagonal source at zero has a tight bound, unlike arbitrary later input.
    assert evolve(src, zero).paths.segments_by_kernel == {"dyson": 2}
    strong = evolve(src, batch(gain=0.1))
    assert strong.paths.segments_by_kernel.get("dyson", 0) == 0
    two_photon = evolve(source((0.0, 0.3, 0.8)), batch(gain=0.002, duration=30000))
    assert two_photon.paths.segments_by_kernel.get("dyson", 0) == 0

    # Continuous parameters remain part of identity even when their lattice
    # representation and final level agree.
    waveform = FluxWaveform(
        np.array([0, 2, 4]),
        np.tile([0.0, 2.0], 2),
        np.array([[0.001], [0], [0.0011], [0]]),
    )
    different_flux = evolve(src, replace(b, flux_waveform=waveform))
    assert different_flux.paths.segments_by_kernel.get("dyson", 0) == 0


def test_flux_tail_identity_ignores_future_samples_but_keeps_prefix_boundary():
    src, b = source(), batch()
    waveform = FluxWaveform(
        np.array([0, 4, 8]),
        np.tile([0.0, 2.0, 50.0, 60.0], 2),
        np.array([[1.0], [0], [0], [0], [1.0], [0], [0], [0.01]]),
    )
    # The probe runs from 7 to 47 ns. Its prefix needs the 50 ns
    # interpolation endpoint, but cannot depend on the 60 ns sample.
    result = evolve(src, replace(b, flux_waveform=waveform))
    assert result.paths.segments_by_kernel == {"t4": 4, "dyson": 2}

    prefix_flux = waveform.flux.copy()
    prefix_flux[4, 0] = 1.01
    unequal_prefix = evolve(
        src, replace(b, flux_waveform=replace(waveform, flux=prefix_flux))
    )
    assert unequal_prefix.paths.segments_by_kernel == {"t4": 4, "rwa": 2}

    # Keep the first sample after probe start in the identity even when
    # equal endpoint values happen to make the interpolated flux flat.
    boundary_times = waveform.time_ns.copy()
    boundary_times[6] = 51.0
    unequal_boundary = evolve(
        src, replace(b, flux_waveform=replace(waveform, time_ns=boundary_times))
    )
    assert unequal_boundary.paths.segments_by_kernel == {"t4": 4, "rwa": 2}


def ordinary_probe(src, b, initial):
    """Evaluate each probe through the model-only public C5 path."""
    track = b.tracks[0]
    n = len(b.points)
    table = SegmentTable(
        np.arange(n),
        track.start_ns,
        track.duration_ns,
        np.zeros((n, 1), dtype=np.int64),
        np.ones(n, dtype=np.int64),
        np.zeros((n, 1), dtype=np.int64),
        track.frequency_ghz[:, None],
        track.amplitude[:, None],
        (2 * np.pi * track.frequency_ghz * track.start_ns)[:, None],
        np.zeros((n, 1), dtype=np.int64),
        np.zeros((n, 1)),
        track.envelopes,
    )
    kernel = DysonKernel()
    rows = np.arange(n)
    keys, first, inverse = np.unique(
        kernel.prepare_keys(table, rows), return_index=True, return_inverse=True
    )
    prepared = kernel.prepare(keys, table, first, src, 1e-3)
    expected = initial.copy()
    mask = kernel.apply(prepared, inverse, table, rows, expected, 1e-3)
    return expected, mask


def test_distinct_repeated_tails_use_their_own_actual_input_with_pruning_budget():
    src, b = source(), batch((7.0, 7.0, 9.0, 9.0))
    result = evolve(src, b)
    assert result.paths.segments_by_kernel == {"t4": 8, "dyson": 4}
    prefix = replace(b, tracks=(), end_ns=b.tracks[0].start_ns)
    initial = evolve(src, prefix).densities
    assert np.max(np.abs(initial[0] - initial[2])) > 0.1
    expected, mask = ordinary_probe(src, b, initial)
    assert not mask.any()
    np.testing.assert_allclose(result.densities, expected, rtol=0, atol=1e-3 / 8)
    assert result.paths.exceeded_by_kernel == {}
    # Source and initial density are evolve-local, not part of a persistent
    # contracted table. The same engine must not reuse another source's input.
    other = ModelSource(np.diag([0.0, 1.0, 0.0]).astype(complex), src.model)
    with closing(
        Engine(1, 100000, kernels=(T4(), DysonKernel(), Fallback(), Fallback("t6")))
    ) as engine:
        engine.evolve(src, (b,), cancel=Event())
        changed = engine.evolve(other, (b,), cancel=Event())
    other_initial = evolve(other, prefix).densities
    other_expected, _ = ordinary_probe(other, b, other_initial)
    np.testing.assert_allclose(changed.densities, other_expected, rtol=0, atol=1e-3 / 8)


def test_flux_tail_population_exceedance_warns_without_retry_or_reroute():
    src, b = source(), batch(gain=0.0005, frequency=0.1)
    with pytest.warns(RuntimeWarning, match="dyson: 2") as caught:
        result = evolve(src, b)
    assert len(caught) == 1
    assert result.paths.segments_by_kernel == {"t4": 4, "dyson": 2}
    assert result.paths.exceeded_by_kernel == {"dyson": 2}
    initial = evolve(src, replace(b, tracks=(), end_ns=b.tracks[0].start_ns)).densities
    expected, mask = ordinary_probe(src, b, initial)
    assert mask.all()
    np.testing.assert_allclose(result.densities, expected, rtol=0, atol=1e-3 / 8)
