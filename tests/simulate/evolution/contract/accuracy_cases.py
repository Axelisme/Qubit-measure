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

from .synthetic_models import synthetic_model


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


def linear_drive_model() -> SecularModel:
    """Return a two-level 0.6 GHz model with a noncircular X-only drive and bath.

    The 2 MHz matrix element keeps omitted counterrotating terms small without
    making them vanish as they would for a circularly polarized drive.
    """
    x = np.array([[0, 0.002], [0.002, 0]], complex)
    return SecularModel(
        (0,),
        "accuracy-linear-drive",
        np.array([0.0, 0.6]),
        np.eye(2, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        np.array([[0.0, 1e-5], [0.0, 0.0]]),
        (np.ones((2, 2)) - np.eye(2)) * 1e-5,
    )


def adiabatic_case() -> AccuracyCase:
    """Build an undriven 8 ns flux ramp with a weak, noncommuting flux term.

    The ramp starts in the DC basis. Its small motion keeps nonadiabatic
    transitions small; the nonzero bath still tests full density coherences.
    """
    model = linear_drive_model()
    source = coherent_source(model)
    expansion = FluxExpansion(
        np.diag(model.energies_ghz).astype(complex),
        (FluxTerm(0, "sin_2pi", model.drives[0].in_phase_ghz * 0.1),),
    )
    source = ModelSource(source.initial_density, source.model, expansion)
    batch = replace(
        timeline(2, 8.0),
        flux_waveform=FluxWaveform(
            np.array([0, 3, 6]),
            np.tile([0.0, 3.0, 8.0], 2),
            np.tile([[0.0], [0.01], [0.02]], (2, 1)),
        ),
    )
    return AccuracyCase("adiabatic_ramp", source, batch, ("adiabatic_map",))


def linear_case(*, shaped: bool) -> AccuracyCase:
    """Build a coherent 24 ns X-only drive, sampled if shaped, else constant.

    Sampled controls exercise gate, Sambe and period playback. Constant controls
    exercise single-tone RWA. Both include decay and off-grid carrier phases.
    """
    batch = timeline(2, 24.0)
    # Real smooth samples satisfy the single-tone gate's slope/curvature
    # domain; X-only polarization still has counterrotating dynamics.
    envelope = (
        EnvelopeSpec(
            "sampled", np.array([0.8, 1.0, 0.8], complex), np.array([0.0, 12.0, 24.0])
        )
        if shaped
        else EnvelopeSpec("constant")
    )
    batch = replace(
        batch, tracks=(drive_track(batch, envelope, frequency=0.6, gain=0.08),)
    )
    families = (
        ("single_tone_gate", "shaped_sambe", "period_table")
        if shaped
        else ("single_tone_rwa",)
    )
    return AccuracyCase(
        "linear_shaped" if shaped else "linear_plateau",
        coherent_source(linear_drive_model()),
        batch,
        families,
    )


def cheap_sampled_case(*, nearby: bool) -> AccuracyCase:
    """Build a smooth sampled three-level case with cheap unexpanded RWA.

    nearby=True uses 0.6/0.6001 GHz carriers; False uses 0.6/0.73 GHz.
    Each case has two 384 ns points, complex coherence and a secular bath.
    Nonflat samples prevent constant-plateau extraction. Return a fresh source
    and history, with the expected post-pause family for the pricing branch.
    """
    name = "cheap_sampled_nearby" if nearby else "cheap_sampled_separated"
    # Circular adjacent transitions retain one three-state component (real
    # dimension 9), with no omitted edges to restore. Both rows previously
    # qualified for the cheap-RWA exemption from common-frame pricing.
    positive = np.diag(np.full(2, 0.001), 1).astype(complex)
    drive = SecularDrive("a", positive + positive.T, 1j * (positive - positive.T))
    model = SecularModel(
        (0,),
        "cheap-sampled-three-level",
        np.array([0.0, 0.6, 1.2]),
        np.eye(3, dtype=complex),
        (drive, replace(drive, name="b")),
        np.array([[0.0, 1e-5, 0.0], [0.0, 0.0, 2e-5], [0.0, 0.0, 0.0]]),
        (np.ones((3, 3)) - np.eye(3)) * 1e-5,
    )
    batch = timeline(2, 384.0)
    envelope = EnvelopeSpec(
        "sampled", np.array([0.8, 1.0, 0.9], complex), np.array([0.0, 192.0, 384.0])
    )
    batch = replace(
        batch,
        tracks=(
            drive_track(batch, envelope, frequency=0.6, gain=0.08, port="a"),
            drive_track(
                batch,
                envelope,
                frequency=0.6001 if nearby else 0.73,
                gain=0.05,
                port="b",
            ),
        ),
    )
    families = ("common_frame",) if nearby else ("dop853",)
    return AccuracyCase(name, coherent_source(model), batch, families)


def joint_case(*, constant: bool) -> AccuracyCase:
    """Build a four-level two-port case with a complete 20-point inventory.

    constant=True uses constant envelopes and permits multitone_rwa/dop853.
    False uses short sampled envelopes and permits only dop853 during the pause.
    Return a fresh source/history with coherence, relaxation and dephasing.
    """
    name = "joint_constant" if constant else "joint_gate_inventory"
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
        np.array([[0, 1e-5, 2e-5, 0], [0, 0, 0, 2e-5], [0, 0, 0, 1e-5], [0, 0, 0, 0]]),
        (np.ones((4, 4)) - np.eye(4)) * 1e-5,
    )
    batch = timeline(20, 1.5)
    shaped = EnvelopeSpec("sampled", np.ones(3, complex), np.array([0.0, 0.75, 1.5]))
    batch = replace(
        batch,
        tracks=(
            drive_track(batch, shaped, frequency=0.6, gain=0.08, port="a"),
            drive_track(batch, shaped, frequency=0.3, gain=0.08, port="b"),
        ),
    )
    if constant:
        batch = replace(
            batch,
            tracks=tuple(
                replace(track, envelopes=(EnvelopeSpec("constant"),))
                for track in batch.tracks
            ),
        )
        families = ("multitone_rwa", "dop853")
    else:
        # Sampled multitone families are paused, not failed density samples.
        families = ("dop853",)
    return AccuracyCase(name, coherent_source(model), batch, families)


def common_frame_case(name: str) -> AccuracyCase:
    """Build nearby_carriers or either cheap_sampled_* pricing input.

    nearby_carriers uses two short constant X-only tones. The two
    cheap_sampled names use cheap_sampled_case with its near/far carriers.
    All return fresh source/history and coherent dissipative models.
    Unknown names raise ValueError.
    """
    if name in ("cheap_sampled_nearby", "cheap_sampled_separated"):
        return cheap_sampled_case(nearby=name == "cheap_sampled_nearby")
    if name != "nearby_carriers":
        raise ValueError(f"unknown common-frame case: {name}")
    constant = EnvelopeSpec("constant")
    model = linear_drive_model()
    batch = timeline(2, 6.5)
    batch = replace(
        batch,
        tracks=(
            drive_track(batch, constant, frequency=0.6, gain=0.08),
            drive_track(batch, constant, frequency=0.63, gain=0.05),
        ),
    )
    families = ("common_frame",)
    return AccuracyCase(name, coherent_source(model), batch, families)


def accuracy_case(name: str) -> AccuracyCase:
    """Build an isolated matrix input; unknown names raise ValueError.

    Names: idle_coherence, weak_plateau, strong_plateau, shaped_complex,
    two_carriers, continuous_flux, joint_gate_inventory, joint_constant, low_carrier,
    linear_shaped, linear_plateau, nearby_carriers, cheap_sampled_nearby,
    cheap_sampled_separated and adiabatic_ramp. Each call
    owns a fresh source/history. The joint gate keeps its 20-point inventory.
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
    elif name == "low_carrier":
        model = linear_drive_model()
        batch = timeline(2, 8.0)
        batch = replace(
            batch, tracks=(drive_track(batch, constant, frequency=0.02, gain=0.2),)
        )
        families = ("slow_envelope",)
    elif name in ("linear_shaped", "linear_plateau"):
        return linear_case(shaped=name == "linear_shaped")
    elif name in ("nearby_carriers", "cheap_sampled_nearby", "cheap_sampled_separated"):
        return common_frame_case(name)
    elif name == "adiabatic_ramp":
        return adiabatic_case()
    elif name in ("joint_gate_inventory", "joint_constant"):
        return joint_case(constant=name == "joint_constant")
    else:
        raise ValueError(f"unknown accuracy case: {name}")
    return AccuracyCase(name, coherent_source(model), batch, families)


def multisegment_case() -> AccuracyCase:
    """Build the accuracy table's complete idle/drive/idle sequence."""
    case = accuracy_case("weak_plateau")
    pulse = replace(
        case.batch.tracks[0],
        start_ns=np.full(2, 0.3),
        duration_ns=np.full(2, 0.9),
        amplitude=np.full(2, 0.05),
    )
    return replace(
        case,
        name="multisegment/final",
        batch=replace(case.batch, tracks=(pulse,), segment_boundaries=((0.7,), (0.7,))),
    )


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
