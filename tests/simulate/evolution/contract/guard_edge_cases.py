"""Importable Engine inputs just inside numerical family admission boundaries.

Each case approaches one numerical threshold, not every conjunct of a guard.
Inputs and margins are explicit here; no production guard/helper is imported.
The paused sampled multitone gate has no numerical-density edge observation.
"""

from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import (
    EnvelopeSpec,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)

from .accuracy_cases import (
    AccuracyCase,
    accuracy_case,
    coherent_source,
    drive_track,
    linear_drive_model,
    timeline,
)

EDGE_FAMILIES = (
    "slow_envelope",
    "dyson",
    "single_tone_rwa",
    "floquet_period",
    "single_tone_gate",
    "shaped_sambe",
    "gate_cache",
    "period_table",
    "common_frame",
    "multitone_rwa",
    "floquet",
)


def _edge_model(family: str) -> SecularModel:
    """Use a stronger bath for the split and spectral-convergence boundaries."""
    model = linear_drive_model()
    if family in ("gate_cache", "floquet_period", "shaped_sambe"):
        return replace(
            model,
            rates_per_ns=np.array([[0.0, 1.0], [0.0, 0.0]]),
            dephasing_per_ns=np.array([[0.0, 0.5], [0.5, 0.0]]),
        )
    return model


def _smooth_rwa_edge(family: str) -> AccuracyCase:
    """Approach normalized slope .025/ns with zero curvature and real samples."""
    batch = timeline(2, 8.0)
    slope = 0.025 * (1 - 1e-6)
    envelope = EnvelopeSpec(
        "sampled",
        np.array([1 - 8 * slope, 1 - 4 * slope, 1], complex),
        np.array([0.0, 4.0, 8.0]),
    )
    tracks = (drive_track(batch, envelope, gain=0.08, frequency=0.6),)
    return AccuracyCase(
        "guard_edge/" + family,
        coherent_source(linear_drive_model()),
        replace(batch, tracks=tracks),
        (family,),
    )


def constant_multitone_guard_case(
    duration_ns: float,
) -> tuple[ModelSource, TimelineBatch]:
    """Build the fixed two-point constant-tone guard scenario for duration in ns.

    The two ports have circular drives; port a also has a diagonal component.
    Both points span 0..duration_ns with phases .17 and 1.37 radians.
    Return raw inputs for admission or rejection; the caller closes the source.
    """
    pump = np.array([[0, 0.1], [0, 0]], complex)
    probe = np.array([[0, 0.0001], [0, 0]], complex)
    model = SecularModel(
        (0,),
        "constant-multitone-guard-edge",
        np.array([0.0, 0.6]),
        np.eye(2, dtype=complex),
        (
            SecularDrive(
                "a",
                pump + pump.conj().T + np.diag([0.0024, -0.0024]),
                1j * (pump - pump.conj().T),
            ),
            SecularDrive("b", probe + probe.conj().T, 1j * (probe - probe.conj().T)),
        ),
        np.array([[0.0, 1e-5], [0.0, 0.0]]),
        np.array([[0.0, 1e-5], [1e-5, 0.0]]),
    )
    batch = timeline(2, duration_ns)
    envelope = EnvelopeSpec("constant")
    batch = replace(
        batch,
        tracks=(
            drive_track(batch, envelope, port="a", frequency=0.6, gain=1.0),
            drive_track(batch, envelope, port="b", frequency=0.63, gain=1.0),
        ),
    )
    return coherent_source(model), batch


def guard_edge_case(family: str) -> AccuracyCase:
    """Build fresh source/history and a complete timeline for one pinned family.

    Unknown families raise ValueError. Units are GHz and ns. Every input has
    nonzero complex coherence, relaxation and dephasing. Most thresholds are
    approached to one part per million from the admitted side. The slow tone
    stays strictly below 30 MHz so current dispatch can reach slow_envelope.
    """
    model = _edge_model(family)
    batch = timeline(2, 8.0)
    frequency, gain = 0.6, 0.08
    envelope = EnvelopeSpec("constant")
    source = None
    if family == "slow_envelope":
        frequency = 0.03 * (1 - 1e-6)
    elif family == "floquet_period":
        batch = timeline(2, 1.0)
        # d*Gamma*(2*pi*gain*||X||/f)^2 / 64^2 = 0.999999.
        # Gamma is the population generator's spectral norm, sqrt(2)/ns.
        gain = frequency * np.sqrt(4096 * (1 - 1e-6) / np.sqrt(2)) / (2 * np.pi * 0.002)
    elif family == "shaped_sambe":
        envelope = EnvelopeSpec(
            "sampled", np.ones(3, complex), np.array([0.0, 4.0, 8.0])
        )
        # Public Engine admission brackets the convergence boundary at gains
        # [71.6674401855469 accepted, 71.66744033813478 rejected]. Both use
        # 7 subdivisions, zero motion and bandwidth far below its limit;
        # the spectral/interpolation estimate <= budget/2 limits this input.
        gain = 71.6674401855469 * (1 - 1e-6)
    elif family == "single_tone_rwa":
        batch = timeline(2, 1.0)
        x = np.diag([frequency, -frequency / 2]).astype(complex)
        model = replace(model, drives=(SecularDrive("drive", x, np.zeros_like(x)),))
        # Both signs give A=diag(gain,gain/2), so kappa2=gain, just below .08.
        # At 1 ns each omitted transfer <=gain**2*f**2<.0058: no restoration.
        gain = 0.08 * (1 - 1e-6)
    elif family == "dyson":
        batch = timeline(2, 1.0)
        # At resonance r = gain*0.002/sqrt(gamma**2 + 1/duration**2).
        # gamma = relaxation/2 + pure dephasing = 1.5e-5 /ns.
        gain = 0.025 * (1 - 1e-6) * np.sqrt(1 + (1.5e-5) ** 2) / 0.002
        coherent = coherent_source(model)
        # A mixed, coherent state leaves the separate population indicator
        # below its threshold while approaching the one-photon r boundary.
        density = 0.999 * np.eye(2) / 2 + 0.001 * coherent.initial_density
        source = ModelSource(density, coherent.model)
    elif family == "single_tone_gate":
        return _smooth_rwa_edge(family)
    elif family == "multitone_rwa":
        case = accuracy_case("joint_constant")
        # Stay just above the 50 MHz absolute-carrier admission boundary.
        tracks = (
            case.batch.tracks[0],
            replace(
                case.batch.tracks[1],
                frequency_ghz=np.full(20, 0.05 * (1 + 1e-6)),
            ),
        )
        return replace(
            case,
            name="guard_edge/" + family,
            batch=replace(case.batch, tracks=tracks),
            families=(family,),
        )
    elif family == "gate_cache":
        envelope = EnvelopeSpec(
            "sampled", np.ones(3, complex), np.array([0.0, 4.0, 8.0])
        )
        # d^3 * Gamma * (gain * peak * ||X||)^2 / 16 = 0.00999999.
        # A stronger bath approaches the split guard without a strong drive.
        gain = np.sqrt(0.01 * (1 - 1e-6) * 16 / 8**3) / 0.002
    elif family == "period_table":
        frequency = 0.03
        envelope = EnvelopeSpec(
            "sampled", np.array([0.8, 0.9, 1], complex), np.array([0.0, 4.0, 8.0])
        )
        # motion = 2*pi*gain*||X||*slope/f^2 = 0.04999995.
        gain = 0.05 * (1 - 1e-6) * frequency**2 / (2 * np.pi * 0.002 * 0.025)
    elif family == "common_frame":
        # Scalar two-level SW algebra gives coherent=0.0017999982 at this
        # fixed scale, one part per million inside the 0.0018 limit.
        # The constant-envelope terms are quadratic, cubic and quartic in
        # scale plus the positive BCH tail. No test-time search is needed.
        scale = 177.19922705562644
        batch = timeline(2, 6.5)
        batch = replace(
            batch,
            tracks=(
                drive_track(batch, envelope, frequency=0.6, gain=0.08 * scale),
                drive_track(batch, envelope, frequency=0.63, gain=0.05 * scale),
            ),
        )
        return AccuracyCase(
            "guard_edge/" + family, coherent_source(model), batch, (family,)
        )
    elif family == "floquet":
        # Sampling extent = energy span + 2*gain*(||X||+||Y||).
        # extent/f = 127.999872, 0.000128 below the 128 limit.
        batch = timeline(2, 1.0)
        gain = 0.1
        frequency = (0.6 + 2 * gain * 0.002) / (128 * (1 - 1e-6))
        batch = replace(
            batch,
            tracks=(
                drive_track(batch, envelope, frequency=frequency, gain=gain),
                drive_track(batch, envelope, frequency=0.11, gain=1e-5),
            ),
        )
        return AccuracyCase(
            "guard_edge/" + family, coherent_source(model), batch, (family,)
        )
    else:
        raise ValueError(f"no numerical guard edge case for {family}")
    batch = replace(
        batch, tracks=(drive_track(batch, envelope, frequency=frequency, gain=gain),)
    )
    return AccuracyCase(
        "guard_edge/" + family,
        source if source is not None else coherent_source(model),
        batch,
        (family,),
    )
