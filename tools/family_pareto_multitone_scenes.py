"""Fixed constant-pair price controls, separate from complete program scans."""

from contextlib import closing
from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import (
    DriveTrack,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)

from tools.family_pareto_inputs import Input, Scene
from tools.family_pareto_scene_builders import integer_fluxonium_input, numeric_builder

MULTITONE_PRICE_SCENES = (
    "mtrwa_c2b_l16_128",
    "mtrwa_c2b_l20_128",
    "mtrwa_c2b_l16_512",
    "mtrwa_star_base",
    "mtrwa_star_action_low",
    "mtrwa_star_action_high",
    "mtrwa_star_dimension_low",
    "mtrwa_star_dimension_high",
    "mtrwa_star_bandwidth_low",
    "mtrwa_star_bandwidth_high",
    "mtrwa_star_duration_low",
    "mtrwa_star_duration_high",
    "mtrwa_star_n20",
    "mtrwa_star_n100",
)
_FAMILIES = ("multitone_rwa", "common_frame", "floquet", "dop853")


def _batch(
    frequencies: tuple[float, float],
    amplitudes: tuple[float, float],
    duration: float,
    repeats: int,
    ports: tuple[str, str],
) -> TimelineBatch:
    tracks = tuple(
        DriveTrack(
            port,
            np.zeros(1),
            np.full(1, duration * repeats),
            np.full(1, frequency),
            np.zeros(1),
            np.full(1, amplitude),
            (EnvelopeSpec("constant"),),
            np.zeros(1, dtype=np.int64),
            np.ones(1, dtype=bool),
        )
        for port, frequency, amplitude in zip(
            ports, frequencies, amplitudes, strict=True
        )
    )
    return TimelineBatch(
        np.arange(1),
        tracks,
        FluxSchedule(np.arange(2), np.zeros(1), np.zeros((1, 1), dtype=np.int64)),
        np.full(1, duration * repeats),
        segment_boundaries=(tuple(duration * k for k in range(1, repeats)),),
    )


def make_multitone_price_inputs(scenario: str | None = None) -> list[Input]:
    """Declare one-support C2b and a one-factor synthetic feature star.

    scenario selects one MULTITONE_PRICE_SCENES name; None returns all fourteen.
    Unknown names raise ValueError before model construction. C2b projects the
    committed integer-fluxonium model to 16/20 levels and uses Pump E10+.5 GHz,
    Stokes E10+.01 GHz, DAC-quantized .02/1e-4 gains and 128/512 ns supports.
    The star holds M=16 with independent three-level blocks. It varies drive
    norm/action, retained block count, restored-edge detuning and duration, one
    at a time. N20/N100 split the base support into consecutive identical rows;
    phases use the same absolute-time convention as Engine. This is a tool-owned
    synthetic context, not a replacement for an experimental FTF model.
    Each Input.build returns a fresh owned source; callers close it. Model and
    numeric errors propagate. This producer neither solves nor evolves.
    """
    if scenario is not None and scenario not in MULTITONE_PRICE_SCENES:
        raise ValueError(f"Unknown multitone price scene: {scenario}")
    names = MULTITONE_PRICE_SCENES if scenario is None else (scenario,)
    return [_c2b(name) if "_c2b_" in name else _star(name) for name in names]


def _c2b(name: str) -> Input:
    size, duration = {
        "mtrwa_c2b_l16_128": (16, 128.0),
        "mtrwa_c2b_l20_128": (20, 128.0),
        "mtrwa_c2b_l16_512": (16, 512.0),
    }[name]
    original = integer_fluxonium_input(
        name,
        lambda _builder, _model: None,
        {"pulse_sequence": "model-only snapshot before hand-declared C2b support"},
        _FAMILIES,
    ).build()
    with closing(original.source):
        full = original.source.model((0,))
        model = replace(
            full,
            identity=f"{full.identity}:leading-{size}",
            energies_ghz=full.energies_ghz[:size],
            eigenvectors=np.eye(size, dtype=complex),
            drives=tuple(
                SecularDrive(
                    d.name, d.in_phase_ghz[:size, :size], d.quadrature_ghz[:size, :size]
                )
                for d in full.drives
            ),
            rates_per_ns=full.rates_per_ns[:size, :size],
            dephasing_per_ns=full.dephasing_per_ns[:size, :size],
        )
        rho = original.source.initial_density[:size, :size].copy()
        rho /= np.trace(rho)
    center = float(model.energies_ghz[1] - model.energies_ghz[0])
    amplitudes = tuple(float(np.floor(g * 32766 + 1e-9) / 32766) for g in (0.02, 1e-4))
    batch = _batch(
        (center + 0.5, center + 0.01),
        (amplitudes[0], amplitudes[1]),
        duration,
        1,
        ("Pump", "Stokes"),
    )
    with closing(ModelSource(rho, lambda _level: model)) as source:
        build = numeric_builder(Scene(name, source, batch, _FAMILIES))
    return Input(
        name,
        build,
        "C2b restored-zero constant-pair price anchor",
        manifest={
            "model": "leading energy-basis projection of INTEGER_MODEL",
            "levels": size,
            "duration_ns": duration,
            "dac_full_scale": 32766,
            "requested_gains": [0.02, 1e-4],
            "carrier_offsets_ghz": [0.5, 0.01],
        },
        category="real",
    )


def _star(name: str) -> Input:
    controls: dict[str, tuple[float, int, float, float, int]] = {
        "base": (0.16, 2, 0.08, 128.0, 1),
        "action_low": (0.14, 2, 0.08, 128.0, 1),
        "action_high": (0.18, 2, 0.08, 128.0, 1),
        "dimension_low": (0.16, 1, 0.08, 128.0, 1),
        "dimension_high": (0.16, 4, 0.08, 128.0, 1),
        "bandwidth_low": (0.16, 2, 0.06, 128.0, 1),
        "bandwidth_high": (0.16, 2, 0.10, 128.0, 1),
        "duration_low": (0.16, 2, 0.08, 32.0, 1),
        "duration_high": (0.16, 2, 0.08, 512.0, 1),
        "n20": (0.16, 2, 0.08, 128.0, 20),
        "n100": (0.16, 2, 0.08, 128.0, 100),
    }
    gain, blocks, detuning, duration, repeats = controls[name.removeprefix("mtrwa_star_")]
    size = 16
    energies = np.arange(size, dtype=float) * 1.5
    drive = np.zeros((size, size), dtype=complex)
    for k in range(4):
        j = 3 * k
        energies[j : j + 3] = 4.5 * k + np.array([0.0, 1.0, 2.0 + detuning])
        if k < blocks:
            drive[j, j + 1] = drive[j + 1, j] = 0.1
            drive[j + 1, j + 2] = drive[j + 2, j + 1] = 0.1
    zero = np.zeros((size, size))
    rates = np.diag(np.full(size - 1, 1e-6), 1)
    model = SecularModel(
        (0,),
        f"mtrwa-star-m16-b{blocks}-detuning{detuning}",
        energies,
        np.eye(size, dtype=complex),
        tuple(SecularDrive(p, drive, zero.astype(complex)) for p in ("pump", "probe")),
        rates,
        zero,
    )
    rho = np.full((size, size), 1 / size, dtype=complex)
    batch = _batch(
        (1.0, 1.005), (gain, gain * 0.01), duration, repeats, ("pump", "probe")
    )
    with closing(ModelSource(rho, lambda _level: model)) as source:
        build = numeric_builder(Scene(name, source, batch, _FAMILIES))
    return Input(
        name,
        build,
        "one-factor restored-edge constant-pair feature star at fixed M=16",
        manifest={
            "model": "synthetic disjoint three-level charge blocks",
            "levels": size,
            "active_blocks": blocks,
            "restored_detuning_ghz": detuning,
            "pump_gain": gain,
            "probe_ratio": 0.01,
            "duration_ns": duration,
            "repeats": repeats,
            "initial_density": "uniform pure superposition",
            "relaxation_per_ns": 1e-6,
        },
    )
