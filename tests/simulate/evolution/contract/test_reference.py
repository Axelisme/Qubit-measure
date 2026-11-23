"""Analytic and composition checks for the independent Lindblad oracle."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
import zcu_tools.simulate.evolution as evolution
from numpy.typing import NDArray

from .reference import (
    SolverTolerances,
    converged_reference,
    density_error,
    reference_timeline,
)

pytestmark = pytest.mark.sim


def test_density_error_includes_high_level_coherences_and_all_points() -> None:
    expected = np.repeat((np.eye(6, dtype=np.complex128) / 6)[None], 2, axis=0)
    actual = expected.copy()
    actual[1, 5, 4] = 0.04 + 0.03j
    actual[1, 4, 5] = 0.04 - 0.03j
    assert density_error(actual, expected) == pytest.approx(0.05)


def doublet_source(
    *, frequency_ghz: float = 0.0, rate_per_ns: float = 0.0
) -> evolution.ModelSource:
    """Return a two-level source initially excited, with an X/Y drive pair."""
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.array([[0, 1j], [-1j, 0]], dtype=np.complex128)
    model = evolution.SecularModel(
        (0,),
        "doublet",
        np.array([0.0, frequency_ghz]),
        np.eye(2, dtype=np.complex128),
        (evolution.SecularDrive("drive", x, y),),
        np.array([[0.0, rate_per_ns], [0.0, 0.0]]),
        np.zeros((2, 2)),
    )
    return evolution.ModelSource(
        np.diag([0, 1]).astype(np.complex128), lambda level: replace(model, level=level)
    )


def idle_batch(end_ns: NDArray[np.float64]) -> evolution.TimelineBatch:
    """Return independent no-drive points at level zero and the given end times."""
    count = len(end_ns)
    return evolution.TimelineBatch(
        np.arange(count),
        (),
        evolution.FluxSchedule(
            np.arange(count + 1), np.zeros(count), np.zeros((count, 1), dtype=np.int64)
        ),
        end_ns,
    )


def drive_track(
    envelope: evolution.EnvelopeSpec,
    *,
    frequency_ghz: float = 0.0,
    phase_rad: float = 0.0,
    amplitude: float = 0.2,
) -> evolution.DriveTrack:
    """Return one active drive over [0,1] ns with a common-origin phase."""
    return evolution.DriveTrack(
        "drive",
        np.array([0.0]),
        np.array([1.0]),
        np.array([frequency_ghz]),
        np.array([phase_rad]),
        np.array([amplitude]),
        (envelope,),
        np.array([0]),
        np.array([True]),
    )


def test_two_tolerance_reference_converges_to_analytic_rabi() -> None:
    source = doublet_source(frequency_ghz=20.0)
    track = drive_track(
        evolution.EnvelopeSpec("constant"), frequency_ghz=20.0, amplitude=5.0
    )
    batch = replace(idle_batch(np.array([0.413])), tracks=(track,))
    result = converged_reference(source, batch)
    angle = 2 * np.pi * 5.0 * 0.413
    vector = np.array(
        [-1j * np.sin(angle), np.exp(-40j * np.pi * 0.413) * np.cos(angle)]
    )
    assert result.convergence_error <= 2e-6
    np.testing.assert_allclose(
        result.density, np.outer(vector, vector.conj())[None], atol=2e-9, rtol=0
    )


def test_two_tolerance_reference_rejects_inaccurate_coarse_solve() -> None:
    source = doublet_source(frequency_ghz=20.0)
    track = drive_track(
        evolution.EnvelopeSpec("constant"), frequency_ghz=20.0, amplitude=5.0
    )
    batch = replace(idle_batch(np.array([0.413])), tracks=(track,))
    with pytest.raises(AssertionError, match="reference did not converge"):
        converged_reference(source, batch, coarse=SolverTolerances(0.1, 0.1))


def test_reference_timeline_matches_resonant_rabi() -> None:
    source = doublet_source(frequency_ghz=2.0)
    time, phase, gain = 0.8, 0.3, 0.2
    track = drive_track(
        evolution.EnvelopeSpec("constant"),
        frequency_ghz=2.0,
        phase_rad=phase,
        amplitude=gain,
    )
    batch = replace(idle_batch(np.array([time])), tracks=(track,))
    angle = 2 * np.pi * gain * time
    vector = np.array(
        [
            -1j * np.exp(1j * phase) * np.sin(angle),
            np.exp(-4j * np.pi * time) * np.cos(angle),
        ]
    )
    np.testing.assert_allclose(
        reference_timeline(source, batch),
        np.outer(vector, vector.conj())[None],
        atol=2e-10,
        rtol=0,
    )


def test_timeline_matches_stitched_flux_and_bath_changes() -> None:
    template = doublet_source(frequency_ghz=0.3, rate_per_ns=0.4).model((0,))
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    expansion = evolution.FluxExpansion(
        np.diag([0.0, 0.3]).astype(np.complex128),
        (evolution.FluxTerm(0, "sin_pi", 0.08 * x),),
    )

    def build(level: evolution.FluxLevel) -> evolution.SecularModel:
        flux = 0.1 * level[0]
        energies, vectors = np.linalg.eigh(
            expansion.static + np.sin(np.pi * flux) * 0.08 * x
        )
        drive = template.drives[0]
        return replace(
            template,
            level=level,
            energies_ghz=energies,
            eigenvectors=vectors,
            drives=(
                evolution.SecularDrive(
                    "drive",
                    vectors.conj().T @ drive.in_phase_ghz @ vectors,
                    vectors.conj().T @ drive.quadrature_ghz @ vectors,
                ),
            ),
            rates_per_ns=template.rates_per_ns * (1 + level[0]),
        )

    initial = np.array([[0.5, 0.3j], [-0.3j, 0.5]])
    source = evolution.ModelSource(initial, build, expansion)
    envelope = evolution.EnvelopeSpec(
        "sampled",
        np.array([0, 0.6 + 0.2j, 0.1j, 0], dtype=np.complex128),
        np.array([0.0, 0.2, 0.5, 1.0]),
    )
    track = replace(
        drive_track(envelope, frequency_ghz=0.27, phase_rad=0.4),
        start_ns=np.array([0.1]),
    )
    batch = evolution.TimelineBatch(
        np.array([17]),
        (track,),
        evolution.FluxSchedule(
            np.array([0, 3]), np.array([0.0, 0.31, 0.7]), np.array([[0], [1], [2]])
        ),
        np.array([1.0]),
        evolution.FluxWaveform(
            np.array([0, 4]),
            np.array([0.0, 0.23, 0.62, 1.0]),
            np.array([[0.0], [0.17], [0.06], [0.2]]),
        ),
        segment_boundaries=((0.31, 0.7),),
    )
    full = reference_timeline(source, batch)
    state = reference_timeline(
        source, replace(batch, end_ns=np.array([0.31]), segment_boundaries=())
    )
    for start, end in [(0.31, 0.7), (0.7, 1.0)]:
        part = replace(
            batch,
            start_ns=np.array([start]),
            end_ns=np.array([end]),
            initial_densities=state,
            segment_boundaries=((0.31, 0.7),) if end >= 0.7 else (),
        )
        state = reference_timeline(source, part)
    np.testing.assert_allclose(full, state, atol=2e-10, rtol=0)
    assert np.max(np.abs(full[0] - initial)) > 0.05


def test_reference_timeline_rejects_explicit_collapse_at_active_level() -> None:
    initial = np.diag([0, 1]).astype(complex)
    with closing(doublet_source(rate_per_ns=0.7)) as secular:
        dc = secular.model((0,))
    jump = np.array([[0, 0.2j], [0, 0]], complex)
    shifted = replace(dc, level=(1,), collapse_operators=(jump,))
    batch = replace(
        idle_batch(np.array([0.3])),
        flux=evolution.FluxSchedule(
            np.array([0, 2]), np.array([0.0, 0.1]), np.array([[0], [1]])
        ),
    )
    models = {dc.level: dc, shifted.level: shifted}
    with (
        closing(evolution.ModelSource(initial, models.__getitem__)) as source,
        pytest.raises(ValueError, match="explicit collapse operators"),
    ):
        reference_timeline(source, batch)


def test_reference_timeline_matches_pure_decay() -> None:
    source = doublet_source(rate_per_ns=0.7)
    times = np.array([0.0, 0.2, 1.3])
    actual = reference_timeline(source, idle_batch(times))
    excited = np.exp(-0.7 * times)
    expected = np.zeros((3, 2, 2), dtype=np.complex128)
    expected[:, 0, 0] = 1 - excited
    expected[:, 1, 1] = excited
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)
