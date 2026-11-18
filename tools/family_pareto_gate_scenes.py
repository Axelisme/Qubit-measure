"""Fixed FTF shaped-gate sequences built at the numeric evolution boundary.

ADR-0072 keeps these curated controls independent of lab settings. ADR-0078
places whole-sequence measurements on TimelineBatch rather than program acquire.
"""

from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, replace

import numpy as np
from zcu_tools.simulate.evolution import (
    DriveTrack,
    EnvelopeSpec,
    FluxLevel,
    FluxSchedule,
    ModelSource,
    TimelineBatch,
)
from zcu_tools.simulators.ftf.config import parse_ftf
from zcu_tools.simulators.ftf.environment.numeric import (
    drive_sources,
    joint_parameters,
    resonators,
)
from zcu_tools.simulators.ftf.physics import JointModel, model_source

from tools.family_pareto_inputs import Input, Scene
from tools.family_pareto_scene_builders import numeric_builder
from tools.family_pareto_settings import FTF_MODEL

# The source spike working files were not committed. These values and complete
# recipes keep this producer independent of those files and mutable lab settings.
# Single-tone X90 calibration: joint ground, bath/dephasing off, evolve 100 ns,
# bisect real gain until the role's maximum-overlap dressed population is .5.
# Calibration used FTF_MODEL at flux .48/.20/.49 Phi0, local4x4x4/retained32.
# This targets population, not gate fidelity. Actual scenes retain joint Gibbs
# density and the fixed model's baths/dephasing, not the calibration state.
# Candidate-tree diagonalization matched the supplied carriers within 1e-9 GHz,
# so the calibrated gains are reused without another evolution calibration.
_X90_CARRIERS = (0.5989899500508908, 0.29771353219577534)
_X90_GAINS = {
    "gaussian": (0.6118698121, 0.7346289760),
    "sampled": (0.7781773395, 0.9345741322),
    "drag": (0.6118698121387076, 0.7346289759718283),
    "flat_top": (0.5570356694528016, 0.6688906298485409),
}
# Local role anharmonicities E2 - 2*E1 + E0 in GHz, F1 then F2.
_DRAG_ALPHA_GHZ = (4.855628336465323, 4.929989930790512)
_PORTS = ("f1_charge", "f2_charge")
# Simultaneous RB, three seeds and two arms at depth500: choose the extremes
# in aligned pairs rather than the complete six-row cross product. The original
# inventory working file was not committed.
_DEPTH500_COUNTS = {
    "min_pairs": (537, 1232, 1436),  # seed42 interleaved: fewest aligned pairs
    "max_pairs": (661, 988, 932),  # seed7 reference: most aligned pairs
}
GATE_SCENES = tuple(
    f"multitone_gate_{shape}_n{count}" for shape in _X90_GAINS for count in (1, 21, 100)
) + tuple(
    f"multitone_depth500_{shape}_{inventory}"
    for shape in ("gaussian", "sampled")
    for inventory in _DEPTH500_COUNTS
)


def _x90_envelope(shape: str, role: int) -> EnvelopeSpec:
    if shape == "gaussian":
        times = np.linspace(0.0, 100.0, 33)
        values = np.exp(-0.5 * ((times - 50.0) / 25.0) ** 2)
        # The calibrated Gaussian retains nonzero endpoints; do not subtract
        # the baseline or substitute another sampling grid.
    elif shape == "sampled":
        times = np.linspace(0.0, 100.0, 32)
        raw = np.random.default_rng(0).normal(size=32)
        kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        smooth = np.convolve(np.pad(raw, (2, 2), mode="edge"), kernel, mode="valid")
        smooth -= float(np.mean(smooth))
        modulation = 0.82 + 0.18 * smooth / float(np.max(np.abs(smooth)))
        values = np.sin(np.pi * times / 100.0) ** 2 * modulation
        values[[0, -1]] = 0.0
    elif shape == "drag":
        times = np.linspace(0.0, 100.0, 33)
        x = times - 50.0
        gaussian = np.exp(-0.5 * (x / 25.0) ** 2)
        derivative = -(x / 25.0**2) * gaussian
        # Q=-dG/dt/alpha, with no 2*pi conversion. Force only Q endpoints
        # to zero, retain G endpoints, then normalize by complex maxabs below.
        quadrature = -derivative / _DRAG_ALPHA_GHZ[role]
        quadrature[[0, -1]] = 0.0
        values = gaussian + 1j * quadrature
    elif shape == "flat_top":
        times = np.linspace(0.0, 100.0, 33)
        distance = np.minimum(times, 100.0 - times)
        values = np.ones_like(times)
        ramp = distance < 25.0
        # Two 25 ns Gaussian ramps, sigma 6.25 ns, 50 ns platform.
        # Unlike CZ, use 33 knots and do not subtract exp(-8) baseline.
        values[ramp] = np.exp(-0.5 * ((distance[ramp] - 25.0) / 6.25) ** 2)
        values[[0, -1]] = 0.0
    else:
        raise ValueError(f"Unknown X90 envelope: {shape}")
    values = values / float(np.max(np.abs(values)))
    return EnvelopeSpec("sampled", values.astype(np.complex128), times)


def _cz_envelope() -> EnvelopeSpec:
    times = np.linspace(0.0, 200.0, 257)
    distance = np.minimum(times, 200.0 - times)
    values = np.ones_like(times)
    ramp = distance < 25.0
    values[ramp] = (
        np.exp(-0.5 * ((distance[ramp] - 25.0) / 6.25) ** 2) - np.exp(-8.0)
    ) / (1.0 - np.exp(-8.0))
    values[[0, -1]] = 0.0
    return EnvelopeSpec("sampled", values.astype(np.complex128), times)


def _track(
    port: str, carrier: float, gain: float, envelope: EnvelopeSpec
) -> DriveTrack:
    return DriveTrack(
        port,
        np.array([0.0]),
        np.array([envelope.sample_ns[-1]]),
        np.array([carrier]),
        np.array([0.0]),
        np.array([gain]),
        (envelope,),
        np.array([0], dtype=np.int64),
        np.array([True]),
    )


def _gate_templates(shape: str) -> dict[str, tuple[DriveTrack, ...]]:
    pair = tuple(
        _track(port, carrier, gain, _x90_envelope(shape, role))
        for role, (port, carrier, gain) in enumerate(
            zip(_PORTS, _X90_CARRIERS, _X90_GAINS[shape], strict=True)
        )
    )
    # Representative CZ coordinates, NOT a calibrated CZ. No conditional phase
    # or fidelity claim: this measures evolution error in an RB-type sequence.
    # At flux .48/.20/.49, max-overlap local |101> and |201> have joint indices
    # 3 and 9 (|201> overlap .9002340156221009 within retained32).
    # Public SecularDrive.in_phase_ghz[9,3], in port order:
    # M1=.0021779567892500477-.041760007149682664j GHz/gain,
    # M2=-.00019802785868649098+.00379697376706631j GHz/gain.
    # |M1+M2|=.03801462905487474; do not sum the separate magnitudes.
    # The actual piecewise-linear 257-knot flat-top area is
    # 165.6538907636908 ns. H/h=gain*M*y*cos(2*pi*f*t) gives RWA angle
    # 2*pi*gain*|M1+M2|*area. Equal gain=1/(|M1+M2|*area) yields 2*pi.
    # No CZ time evolution or optimization was used to select these values.
    cz = tuple(
        _track(port, 5.443720064991563, 0.1587989411334408, _cz_envelope())
        for port in _PORTS
    )
    return {"pair": pair, "tail_f1": pair[:1], "tail_f2": pair[1:], "cz": cz}


def _gate_order(inventory: str) -> list[str]:
    pairs, tails, cz = _DEPTH500_COUNTS[inventory]
    # Alternate the tail roles, then use one fixed permutation for both shapes.
    order = (
        ["pair"] * pairs
        + ["tail_f1" if index % 2 == 0 else "tail_f2" for index in range(tails)]
        + ["cz"] * cz
    )
    return [str(gate) for gate in np.random.default_rng(0).permutation(order)]


def make_gate_inputs(scenario: str | None = None) -> list[Input]:
    """Declare fixed-FTF repeated pairs and depth500 three-gate sequences.

    scenario selects one GATE_SCENES name before any model work; None builds all.
    Unknown names raise ValueError. ModelSource uses the fixed model's Gibbs
    density, baths and dephasing at flux .48/.20/.49 Phi0, local4x4x4/retained32.
    X90 pairs/tails last 100 ns; the representative, uncalibrated CZ lasts
    200 ns. Pulse recipes and calibration provenance are recorded here, without
    importing lab definitions or reading user settings. Construction never
    evolves density, solves references, or writes an index/cache. Each returned
    Input.build creates a fresh caller-owned source that must be closed.
    """
    if scenario is not None and scenario not in GATE_SCENES:
        raise ValueError(
            f"Unknown gate scene: {scenario}. Legal names: {', '.join(GATE_SCENES)}"
        )
    names = GATE_SCENES if scenario is None else (scenario,)
    settings = parse_ftf(FTF_MODEL)
    joint = JointModel(
        joint_parameters(settings),
        np.array(settings.dc_bias.offset_phi0.as_tuple()),
        temperature_k=settings.chip.temperature_k,
    )
    inputs = []
    with closing(
        model_source(
            joint, drive_sources(settings, joint), resonators=resonators(settings)
        )
    ) as source:
        for name in names:
            shape, suffix = (
                name.removeprefix("multitone_gate_").rsplit("_n", 1)
                if name.startswith("multitone_gate_")
                else name.removeprefix("multitone_depth500_").split("_", 1)
            )
            order = (
                ["pair"] * int(suffix)
                if name.startswith("multitone_gate_")
                else _gate_order(suffix)
            )
            templates = _gate_templates(shape)
            batch = numeric_gate_sequence(source, templates, order, level=(0, 0, 0))
            families = ("multitone_gate", "dop853")
            scene = Scene(name, source, batch, families)
            inputs.append(
                Input(
                    name,
                    numeric_builder(scene),
                    "fixed-model shaped pair reuse and RB-type gate-count sequence",
                    category="real",
                    companions_by_target={"multitone_gate": ("dop853",)},
                    manifest={
                        # Acquisition capture is not part of numeric provenance.
                        "model_settings": settings.model_dump(
                            mode="json", exclude={"numerics": {"debug_capture"}}
                        ),
                        "measurement_layer": "kernel"
                        if name.startswith("multitone_gate_")
                        else "sequence",
                        "pulse_sequence": "contiguous gates, absolute carrier phase, ordered F1 then F2 tracks",
                        "shape": shape,
                        "gate_order": order,
                        "gate_templates": {
                            gate: [asdict(track) for track in tracks]
                            for gate, tracks in templates.items()
                        },
                        "inventory_source": "three seeds/two arms simultaneous RB depth500; min=seed42 interleaved, max=seed7 reference; original inventory not committed",
                        "calibration_source": "fixed-model single-tone joint-ground population .5 at 100ns, baths/dephasing off; original spike not committed; CZ only RWA matrix-element area, not calibrated phase/fidelity",
                    },
                )
            )
    return inputs


def numeric_gate_sequence(
    source: ModelSource,
    gates: Mapping[str, tuple[DriveTrack, ...]],
    order: Sequence[str],
    *,
    level: FluxLevel,
) -> TimelineBatch:
    """Arrange named simultaneous gate templates into one numeric sequence.

    source supplies the allowed drive ports at the constant integer flux level.
    gates maps each caller label to ordered one-point DriveTracks, all active,
    starting at zero with the same positive duration. Tracks carry the role's
    port, carrier in GHz, real gain, phase at absolute t=0 in radians and local
    EnvelopeSpecs. order lists labels in playback order, allowing repeats.
    Return a one-point TimelineBatch with contiguous supports in ns; preserve
    template track order, carrier, phase and pulse-local envelope coordinates.
    No evolution runs and the caller retains source ownership. Missing labels
    raise KeyError. Empty sequences/templates, unknown ports, multi-point or
    nonaligned templates raise ValueError. Model construction errors propagate.
    """
    if not order:
        raise ValueError("gate sequence must not be empty")
    ports = {drive.name for drive in source.model(level).drives}
    start = 0.0
    tracks: list[DriveTrack] = []
    for name in order:
        template = gates[name]
        duration = _gate_duration(template, ports)
        tracks.extend(replace(track, start_ns=np.array([start])) for track in template)
        start += duration
    return TimelineBatch(
        np.array([0], dtype=np.int64),
        tuple(tracks),
        FluxSchedule(
            np.array([0, 1], dtype=np.int64),
            np.array([0.0]),
            np.array([level], dtype=np.int64),
        ),
        np.array([start]),
    )


def _gate_duration(tracks: tuple[DriveTrack, ...], ports: set[str]) -> float:
    if not tracks:
        raise ValueError("gate templates must contain tracks")
    if any(track.start_ns.shape != (1,) for track in tracks):
        raise ValueError("gate templates must have one point")
    duration = float(tracks[0].duration_ns[0])
    if duration <= 0 or any(
        not track.active[0]
        or track.start_ns[0] != 0
        or track.duration_ns[0] != duration
        for track in tracks
    ):
        raise ValueError("gate templates must be active and aligned at zero")
    if any(track.port not in ports for track in tracks):
        raise ValueError("gate template has an unknown drive port")
    return duration
