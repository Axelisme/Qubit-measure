"""Small coherent Engine inputs shared by the accuracy matrix and consumers."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

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
from zcu_tools.simulate.fluxonium import (
    DispersiveResonator,
    FluxoniumNoise,
    SingleFluxoniumModel,
    calculate_oscillator_operators,
)

from .synthetic_models import synthetic_model


@dataclass(frozen=True)
class AccuracyCase:
    """Named physical input and independently pinned accuracy targets.

    source owns model/history state; batch contains all points. families lists
    independent pinned targets. idle_family names the undriven companion needed
    alongside each target; None means no idle companion. These declarations are
    test expectations, not per-segment Engine routing.
    """

    name: str
    source: ModelSource
    batch: TimelineBatch
    families: tuple[str, ...]
    idle_family: str | None = None


DISSIPATIVE_DC_CASES = ("dephasing_pump", "raman_decay", "raman_weak_legs")
OPENCL_CASES = (
    "sampled_gaussian_9",
    "sampled_gaussian_960",
    "sampled_cosine",
    "sampled_gaussian_960_long",
)


def dissipative_dc_case(name: str) -> AccuracyCase:
    """Build zero-carrier bath-assisted transfer from an initial ground state.

    Dephasing pump uses pair dephasing 0.1/ns for 1000 ns. Raman decay uses
    a 0.02/ns jump for 500 ns; weak legs use 0.0002/ns for 12500 ns.
    The latter is within a 0.01/ns rate ceiling; the former two exceed it.
    """
    if name == "dephasing_pump":
        energies = [0.0, 0.1, 0.5]
        edges = ((0, 1, 0.001),)
        jumps = ()
        phi, duration = 0.1, 1000.0
    elif name in ("raman_decay", "raman_weak_legs"):
        energies = [0.0, 0.0, 0.02, 0.2, 0.7]
        weak = name == "raman_weak_legs"
        coupling = 0.002 if weak else 0.01
        edges = ((0, 3, coupling), (1, 3, coupling))
        jumps = ((2, 3, 0.0002 if weak else 0.02),)
        phi, duration = 0.0, 12500.0 if weak else 500.0
    else:
        raise ValueError(f"unknown dissipative DC case: {name}")
    size = len(energies)
    x = np.zeros((size, size), dtype=complex)
    rates = np.zeros((size, size))
    for a, b, coupling in edges:
        x[a, b] = x[b, a] = coupling
    for target, source, rate in jumps:
        rates[target, source] = rate
    model = SecularModel(
        (0,),
        "accuracy-" + name,
        np.array(energies),
        np.eye(size, dtype=complex),
        (SecularDrive("drive", x, np.zeros_like(x)),),
        rates,
        phi * (np.ones((size, size)) - np.eye(size)),
    )
    initial = np.zeros((size, size), dtype=complex)
    initial[0, 0] = 1
    batch = timeline(1, duration)
    pulse = replace(
        drive_track(batch, EnvelopeSpec("constant"), frequency=0.0, gain=1.0),
        phase_rad=np.zeros(1),
    )
    return AccuracyCase(
        name,
        ModelSource(initial, lambda level: replace(model, level=level)),
        replace(batch, tracks=(pulse,)),
        ("slow_envelope",),
    )


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


def single_tone_sideband_case() -> AccuracyCase:
    """Build the synthetic 600 ns single-tone reset with five cavity lifetimes idle.

    Full native cutoff 30 is dressed before retaining six parent levels times
    three photons. Flux is 0.40, fr=6 GHz, g=0.08 GHz, kappa/(2*pi)=5 MHz.
    The charge tone drives dressed |e,0> to |g,1> at g_sb=1 MHz; the initial
    state is |e,0>. Intrinsic capacitive Q=1e6 loss remains active at zero K.
    Plateau targets are floquet_period and t6, each with the t4 idle companion.
    """
    circuit = SingleFluxoniumModel(
        (4.0, 1.0, 1.0), 0.40, cutoff=30, retained_levels=6, temperature_k=0
    )
    kappa = 2 * np.pi * 0.005
    native = calculate_oscillator_operators(circuit.params, cutoff=circuit.cutoff)
    joint = circuit.joint_resonator_model(
        (0,),
        DispersiveResonator(6.0, 0.08, kappa),
        drive_operators_ghz={"drive": native.charge},
        noise=FluxoniumNoise(q_cap=1e6),
    )
    model = joint.model
    excited = joint.label_to_index[(1, 0)]
    photon = joint.label_to_index[(0, 1)]
    initial = np.zeros_like(model.eigenvectors)
    initial[excited, excited] = 1
    source = ModelSource(initial, lambda level: replace(model, level=level))
    batch = timeline(1, 600.0 + 5 / kappa)
    track = drive_track(
        batch,
        EnvelopeSpec("constant"),
        frequency=float(model.energies_ghz[photon] - model.energies_ghz[excited]),
        gain=2 * 0.001 / abs(model.drives[0].in_phase_ghz[photon, excited]),
    )
    batch = replace(batch, tracks=(replace(track, duration_ns=np.array([600.0])),))
    return AccuracyCase(
        "single_tone_sideband",
        source,
        batch,
        ("floquet_period", "t6"),
        idle_family="t4",
    )


def _opencl_case(name: str) -> AccuracyCase:
    large_model = name == "sampled_gaussian_960"
    long = name == "sampled_gaussian_960_long"
    cosine = name == "sampled_cosine"
    model = synthetic_model(21 if large_model else 3)
    duration = 1000.0 if long else 100.0
    count = 1 if large_model or long else 2
    knots = 960 if large_model or long else 41 if cosine else 9
    times = np.linspace(0.0, duration, knots, endpoint=not large_model)
    samples = (
        np.sin(np.pi * times / duration) ** 2
        if cosine
        else np.exp(-0.5 * ((times - duration / 2) / (duration / 6)) ** 2)
    )
    envelope = EnvelopeSpec("sampled", samples.astype(np.complex128), times)
    batch = timeline(count, duration)
    tracks = (
        drive_track(batch, envelope, frequency=0.129, gain=0.003),
        drive_track(batch, envelope, frequency=0.17, gain=-0.007),
    )
    # The dense case has a CPU idle tail after the GPU returns a density.
    if large_model:
        batch = replace(batch, end_ns=batch.end_ns + 2.3)
    return AccuracyCase(
        name,
        coherent_source(model),
        replace(batch, tracks=tracks),
        ("dop853",),
        idle_family="t4" if large_model else None,
    )


def accuracy_case(name: str) -> AccuracyCase:
    """Build an isolated matrix input; unknown names raise ValueError.

    Names: idle_coherence, weak_plateau, strong_plateau, shaped_complex,
    two_carriers, continuous_flux, joint_gate_inventory, joint_constant, low_carrier,
    linear_shaped, linear_plateau, nearby_carriers, cheap_sampled_nearby,
    cheap_sampled_separated, adiabatic_ramp, single_tone_sideband,
    dephasing_pump, raman_decay, raman_weak_legs, and the OPENCL_CASES.
    Each call
    owns a fresh source/history. The joint gate keeps its 20-point inventory.
    """
    if name in DISSIPATIVE_DC_CASES:
        return dissipative_dc_case(name)
    if name in OPENCL_CASES:
        return _opencl_case(name)
    specialized: dict[str, Callable[[], AccuracyCase]] = {
        "linear_shaped": partial(linear_case, shaped=True),
        "linear_plateau": partial(linear_case, shaped=False),
        "nearby_carriers": partial(common_frame_case, "nearby_carriers"),
        "cheap_sampled_nearby": partial(common_frame_case, "cheap_sampled_nearby"),
        "cheap_sampled_separated": partial(
            common_frame_case, "cheap_sampled_separated"
        ),
        "adiabatic_ramp": adiabatic_case,
        "joint_gate_inventory": partial(joint_case, constant=False),
        "joint_constant": partial(joint_case, constant=True),
        "single_tone_sideband": single_tone_sideband_case,
    }
    if name in specialized:
        return specialized[name]()
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
