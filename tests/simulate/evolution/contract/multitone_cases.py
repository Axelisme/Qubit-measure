"""Test-owned constant-pair inputs through public physical/evolution builders.

The A16 parameters are fixed before fresh reference results are observed.
No tools, lab settings, saved matrices or reference results are imported.
"""

from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import (
    DriveSource,
    DriveTrack,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulate.fluxonium import (
    DispersiveResonator,
    FluxoniumNoise,
    SingleFluxoniumModel,
)


def short_pair_case() -> tuple[ModelSource, TimelineBatch]:
    """Return ground-state, distinct-port circular tones on a two-level model.

    All nonzero Fourier couplings lie inside the retained component. Omitted
    transfer, phase and secular accumulation are therefore zero. This is an
    admission input, not an accuracy reference or a physical-workload proxy.
    """
    positive = np.array([[0.0, 0.1], [0.0, 0.0]], dtype=np.complex128)
    x = positive + positive.conj().T
    y = 1j * (positive - positive.conj().T)
    model = SecularModel(
        (0,),
        "test-owned-short-constant-pair",
        np.array([0.0, 0.6]),
        np.eye(2, dtype=np.complex128),
        tuple(SecularDrive(port, x, y) for port in ("a", "b")),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    ground = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    source = ModelSource(ground, lambda level: replace(model, level=level))
    tracks = tuple(
        DriveTrack(
            port,
            np.array([0.0]),
            np.array([8.0]),
            np.array([frequency]),
            np.array([0.0]),
            np.array([1.0]),
            (EnvelopeSpec("constant"),),
            np.array([0], dtype=np.int64),
            np.array([True]),
        )
        for port, frequency in (("a", 0.6), ("b", 0.63))
    )
    batch = TimelineBatch(
        np.array([0], dtype=np.int64),
        tracks,
        FluxSchedule(
            np.array([0, 1], dtype=np.int64),
            np.array([0.0]),
            np.array([[0]], dtype=np.int64),
        ),
        np.array([8.0]),
    )
    return source, batch


def accumulated_pair_case() -> tuple[ModelSource, TimelineBatch]:
    """Return fixed A16, ground-state, 1600 ns distinct-port constant tones.

    The 80-oscillator/20-level public model is reduced to its lowest 16 energy
    coordinates, including the corresponding bath submatrices. The ground
    density is explicitly supplied rather than using the model Gibbs state.
    GHz, ns, rad and inverse-ns units follow the public evolution records.
    No result-dependent tuning, initial-state substitution or persisted matrices.
    """
    physical = SingleFluxoniumModel(
        (6.78, 1.47, 0.22),
        0.0,
        cutoff=80,
        retained_levels=20,
        temperature_k=0.020,
    )
    charge = physical.charge_operator()
    full = physical.secular_model(
        (0,),
        (DriveSource("Pump", charge), DriveSource("Stokes", charge)),
        noise=FluxoniumNoise(q_cap=100000.0, q_ind=None, xqp=0.0, aphi=0.0),
        resonator=DispersiveResonator(
            frequency_ghz=7.46,
            g_ghz=0.10,
            kappa_per_ns=2 * np.pi * 0.015,
            thermal_occupation=0.0,
        ),
    )
    size = 16
    model = SecularModel(
        (0,),
        "test-owned-accumulated-pair-a16",
        full.energies_ghz[:size],
        np.eye(size, dtype=np.complex128),
        tuple(
            SecularDrive(
                drive.name,
                drive.in_phase_ghz[:size, :size],
                drive.quadrature_ghz[:size, :size],
            )
            for drive in full.drives
        ),
        full.rates_per_ns[:size, :size],
        full.dephasing_per_ns[:size, :size],
    )
    ground = np.zeros((size, size), dtype=np.complex128)
    ground[0, 0] = 1.0
    source = ModelSource(ground, lambda level: replace(model, level=level))
    tracks = tuple(
        DriveTrack(
            port,
            np.array([0.0]),
            np.array([1600.0]),
            np.array([frequency]),
            np.array([0.0]),
            np.array([amplitude]),
            (EnvelopeSpec("constant"),),
            np.array([0], dtype=np.int64),
            np.array([True]),
        )
        for port, frequency, amplitude in (
            ("Pump", 4.119715413356692, 0.06998107794665202),
            ("Stokes", 4.079715413356692, 0.006988951962400049),
        )
    )
    batch = TimelineBatch(
        np.array([0], dtype=np.int64),
        tracks,
        FluxSchedule(
            np.array([0, 1], dtype=np.int64),
            np.array([0.0]),
            np.array([[0]], dtype=np.int64),
        ),
        np.array([1600.0]),
    )
    return source, batch


def restored_dimension_case() -> tuple[ModelSource, TimelineBatch]:
    """Return ground-state, 128 ns constant tones on a 17-level chain.

    Energies are spaced by 0.3 GHz. Distinct ports have identical adjacent
    X couplings of 0.02 GHz/gain and zero Y couplings. Unity gains at 0.36
    and 0.37 GHz drive both ports with zero phase. Baths are zero.
    This fixed admission input checks final-mask diagnostics after restoration,
    not full-density accuracy or a physical-workload proxy.
    """
    size = 17
    adjacent = np.full(size - 1, 0.02, dtype=np.complex128)
    x = np.diag(adjacent, 1) + np.diag(adjacent, -1)
    y = np.zeros((size, size), dtype=np.complex128)
    model = SecularModel(
        (0,),
        "test-owned-restored-dimension-pair",
        0.3 * np.arange(size, dtype=np.float64),
        np.eye(size, dtype=np.complex128),
        tuple(SecularDrive(port, x, y) for port in ("a", "b")),
        np.zeros((size, size)),
        np.zeros((size, size)),
    )
    ground = np.zeros((size, size), dtype=np.complex128)
    ground[0, 0] = 1.0
    source = ModelSource(ground, lambda level: replace(model, level=level))
    tracks = tuple(
        DriveTrack(
            port,
            np.array([0.0]),
            np.array([128.0]),
            np.array([frequency]),
            np.array([0.0]),
            np.array([1.0]),
            (EnvelopeSpec("constant"),),
            np.array([0], dtype=np.int64),
            np.array([True]),
        )
        for port, frequency in (("a", 0.36), ("b", 0.37))
    )
    batch = TimelineBatch(
        np.array([0], dtype=np.int64),
        tracks,
        FluxSchedule(
            np.array([0, 1], dtype=np.int64),
            np.array([0.0]),
            np.array([[0]], dtype=np.int64),
        ),
        np.array([128.0]),
    )
    return source, batch
