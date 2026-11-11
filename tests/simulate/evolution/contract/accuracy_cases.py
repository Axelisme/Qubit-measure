"""Small coherent Engine inputs shared by the accuracy matrix and consumers."""

from dataclasses import dataclass, replace

import numpy as np
from zcu_tools.simulate.evolution import (
    DriveTrack,
    EnvelopeSpec,
    FluxExpansion,
    FluxSchedule,
    FluxTerm,
    FluxWaveform,
    ModelSource,
    RoutingHistory,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)

from .kernel_suite import synthetic_model


@dataclass(frozen=True)
class AccuracyCase:
    """Named physical input and families expected to accept its complete timeline.

    source owns model/history state; batch contains all points. families lists
    independently pinnable routes, not a per-segment algorithm preference.
    """

    name: str
    source: ModelSource
    batch: TimelineBatch
    families: tuple[str, ...]


def coherent_source(model: SecularModel) -> ModelSource:
    """Build a source with full complex coherence and the supplied constant bath."""
    vector = np.exp(0.7j * np.arange(len(model.energies_ghz)))
    vector /= np.linalg.norm(vector)
    return ModelSource(
        np.outer(vector, vector.conj()), lambda level: replace(model, level=level)
    )


def timeline(count: int, duration: float) -> TimelineBatch:
    """Build count independent idle timelines from zero to duration ns."""
    return TimelineBatch(
        np.arange(count),
        (),
        FluxSchedule(
            np.arange(count + 1), np.zeros(count), np.zeros((count, 1), dtype=np.int64)
        ),
        np.full(count, duration),
    )


def drive_track(
    batch: TimelineBatch,
    envelope: EnvelopeSpec,
    *,
    frequency: float = 0.13,
    gain: float = 0.05,
    port: str = "drive",
) -> DriveTrack:
    """Add a full-support tone; frequency is GHz, gain is dimensionless.

    Each point has a distinct off-grid phase to exercise density coherences.
    """
    count = len(batch.points)
    return DriveTrack(
        port,
        np.zeros(count),
        batch.end_ns,
        np.full(count, frequency),
        np.linspace(0.17, 1.37, count),
        np.full(count, gain),
        (envelope,),
        np.zeros(count, dtype=np.int64),
        np.ones(count, dtype=bool),
    )


def accuracy_case(name: str) -> AccuracyCase:
    """Build an isolated matrix input; unknown names raise ValueError.

    Names: idle_coherence, weak_plateau, strong_plateau, shaped_complex,
    two_carriers, continuous_flux and joint_gate_inventory. Each call owns a
    fresh source/history. The joint gate case keeps the full 20-point inventory.
    """
    model = synthetic_model()
    batch = timeline(2, 1.5)
    constant = EnvelopeSpec("constant")
    shaped = EnvelopeSpec(
        "sampled",
        np.array([0.2 + 0.1j, 1.0 - 0.3j, 0.4 + 0.2j]),
        np.array([0.0, 0.6, 1.5]),
    )
    if name == "idle_coherence":
        families = ("t4",)
    elif name == "weak_plateau":
        batch = replace(batch, tracks=(drive_track(batch, constant, gain=4e-5),))
        families = ("dyson", "floquet_period", "t6")
    elif name == "strong_plateau":
        batch = replace(
            batch, tracks=(drive_track(batch, constant, frequency=0.7, gain=0.08),)
        )
        families = ("floquet_period", "t6")
    elif name == "shaped_complex":
        batch = replace(batch, tracks=(drive_track(batch, shaped),))
        families = ("gate_cache", "dop853")
    elif name == "two_carriers":
        model = replace(
            model,
            drives=tuple(
                replace(
                    d,
                    in_phase_ghz=d.in_phase_ghz * 0.001,
                    quadrature_ghz=d.quadrature_ghz * 0.001,
                )
                for d in model.drives
            ),
        )
        batch = replace(
            batch,
            tracks=(
                drive_track(batch, constant, frequency=3.7, gain=0.1),
                drive_track(batch, constant, frequency=0.11, gain=0.002),
            ),
        )
        families = ("floquet", "dop853")
    elif name == "continuous_flux":
        source = coherent_source(model)
        expansion = FluxExpansion(
            np.diag(model.energies_ghz).astype(complex),
            (FluxTerm(0, "sin_2pi", model.drives[0].in_phase_ghz * 0.01),),
        )
        source = ModelSource(source.initial_density, source.model, expansion)
        waveform = FluxWaveform(
            np.array([0, 3, 6]),
            np.tile([0.0, 0.7, 1.5], 2),
            np.tile([[0.0], [0.1], [-0.05]], (2, 1)),
        )
        batch = replace(
            batch, tracks=(drive_track(batch, shaped),), flux_waveform=waveform
        )
        return AccuracyCase(name, source, batch, ("flux_dop853",))
    elif name == "joint_gate_inventory":
        a, b = np.zeros((4, 4), complex), np.zeros((4, 4), complex)
        a[0, 2] = a[1, 3] = 0.002
        b[0, 1] = b[2, 3] = 0.002
        model = SecularModel(
            (0,),
            "accuracy-joint-gate",
            np.array([0.0, 0.3, 0.6, 0.9]),
            np.eye(4, dtype=complex),
            (
                SecularDrive("a", a + a.T, np.zeros_like(a)),
                SecularDrive("b", b + b.T, np.zeros_like(b)),
            ),
            np.array(
                [[0, 1e-5, 2e-5, 0], [0, 0, 0, 2e-5], [0, 0, 0, 1e-5], [0, 0, 0, 0]]
            ),
            (np.ones((4, 4)) - np.eye(4)) * 1e-5,
        )
        batch = timeline(20, 1.5)
        shaped = EnvelopeSpec(
            "sampled", np.ones(3, complex), np.array([0.0, 0.75, 1.5])
        )
        batch = replace(
            batch,
            tracks=(
                drive_track(batch, shaped, frequency=0.6, gain=0.08, port="a"),
                drive_track(batch, shaped, frequency=0.3, gain=0.08, port="b"),
            ),
        )
        families = ("multitone_gate", "multitone_rwa", "dop853")
    else:
        raise ValueError(f"unknown accuracy case: {name}")
    return AccuracyCase(name, coherent_source(model), batch, families)


def history_case(count: int, history: RoutingHistory) -> AccuracyCase:
    """Build count shaped 24 ns gates sharing caller-owned routing history.

    Two points are cold Sambe; an eight-point inventory makes later two-point
    inputs choose single_tone_gate. families is empty because this case checks
    default dispatch rather than independent pinned routes.
    """
    x = np.array([[0, 0.002], [0.002, 0]], complex)
    model = SecularModel(
        (0,),
        "accuracy-history",
        np.array([0.0, 0.6]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        np.array([[0.0, 1e-5], [0.0, 0.0]]),
        (np.ones((2, 2)) - np.eye(2)) * 1e-5,
    )
    src = coherent_source(model)
    source = ModelSource(src.initial_density, src.model, history=history)
    batch = timeline(count, 24.0)
    envelope = EnvelopeSpec(
        "sampled", np.array([0.8, 1.0, 0.8], complex), np.array([0.0, 12.0, 24.0])
    )
    batch = replace(
        batch, tracks=(drive_track(batch, envelope, frequency=0.6, gain=0.08),)
    )
    return AccuracyCase("history_gate", source, batch, ())
