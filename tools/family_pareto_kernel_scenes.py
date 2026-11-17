"""Single-segment kernel controls over a fixed public fluxonium model.

Unlike complete compiled sequences, these hand-declared batches contain no
frontend gaps or readout lead. Each point is one simultaneous support interval.
"""

from contextlib import closing
from dataclasses import asdict

import numpy as np
from zcu_tools.simulate.evolution import (
    DriveSource,
    DriveTrack,
    EnvelopeSpec,
    FluxSchedule,
    TimelineBatch,
)
from zcu_tools.simulate.fluxonium import SingleFluxoniumModel

from tools.family_pareto_inputs import Input, Scene
from tools.family_pareto_scene_builders import numeric_builder

KERNEL_SCENES = (
    "kernel_single_constant",
    "kernel_single_sampled",
    "kernel_multitone_sampled",
)


def make_kernel_inputs(scenario: str | None = None) -> list[Input]:
    """Declare single-support dop853 workloads, optionally selecting one by name.

    Use fixed EJ/EC/EL=4/1/1 GHz, DC .5 Phi0, native cutoff40, 21 retained
    levels and temperature .03 K through SingleFluxoniumModel. No loss model is
    enabled. Each scene has four gain points .01/.04/.07/.1, 100 ns support,
    charge drive coefficient .1 GHz/gain and no readout/idle spans. Sampled
    envelopes use nine Gaussian knots with sigma20 ns; multitone adds a carrier
    at 1.25 times DC E1-E0 on the same charge port, starting and ending together.
    These are illustrative uncalibrated controls, not experimental parameters.

    Return Input declarations with fresh owned numeric sources from build;
    callers close those sources. Unknown names raise ValueError before any
    model construction. Model construction and numeric validation errors propagate.
    """
    if scenario is not None and scenario not in KERNEL_SCENES:
        raise ValueError(
            f"Unknown kernel scene: {scenario}. Legal names: {', '.join(KERNEL_SCENES)}"
        )
    names = KERNEL_SCENES if scenario is None else (scenario,)
    model = SingleFluxoniumModel(
        (4.0, 1.0, 1.0), 0.5, cutoff=40, retained_levels=21, temperature_k=0.03
    )
    source = model.model_source((DriveSource("charge", 0.1 * model.charge_operator()),))
    inputs = []
    with closing(source):
        center = float(model.energies_ghz[1] - model.energies_ghz[0])
        times = np.linspace(0.0, 100.0, 9)
        sampled = EnvelopeSpec(
            "sampled",
            np.exp(-0.5 * ((times - 50.0) / 20.0) ** 2).astype(complex),
            times,
        )
        for name in names:
            envelope = (
                EnvelopeSpec("constant")
                if name == "kernel_single_constant"
                else sampled
            )
            carriers = (
                (center, center * 1.25)
                if name == "kernel_multitone_sampled"
                else (center,)
            )
            tracks = tuple(
                DriveTrack(
                    "charge",
                    np.zeros(4),
                    np.full(4, 100.0),
                    np.full(4, carrier),
                    np.zeros(4),
                    np.array([0.01, 0.04, 0.07, 0.1]),
                    (envelope,),
                    np.zeros(4, dtype=np.int64),
                    np.ones(4, dtype=bool),
                )
                for carrier in carriers
            )
            batch = TimelineBatch(
                np.arange(4),
                tracks,
                FluxSchedule(
                    np.arange(5), np.zeros(4), np.zeros((4, 1), dtype=np.int64)
                ),
                np.full(4, 100.0),
            )
            inputs.append(
                Input(
                    name,
                    numeric_builder(Scene(name, source, batch, ("dop853",))),
                    "one simultaneous kernel support per point, without frontend compilation",
                    category="real",
                    expected_routes={"dop853": {"dop853": (4, 400.0)}},
                    manifest={
                        "model": {
                            "params_ghz": [4.0, 1.0, 1.0],
                            "dc_flux_phi0": 0.5,
                            "cutoff": 40,
                            "retained_levels": 21,
                            "temperature_k": 0.03,
                            "noise": "disabled",
                        },
                        "pulse_sequence": "one simultaneous support interval per point",
                        "point_gains": [0.01, 0.04, 0.07, 0.1],
                        "carriers_ghz": carriers,
                        "envelope": asdict(envelope),
                        "duration_ns": 100.0,
                        "drive_coefficient_ghz_per_gain": 0.1,
                        "measurement_layer": "kernel",
                    },
                )
            )
    return inputs
