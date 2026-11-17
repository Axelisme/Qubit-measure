"""Self-contained physical inputs for finite constant-RWA sweeps.

Only input construction lives here. Consumers observe the public diagnostic and
Engine paths, and use interaction_reference for the full-density comparison.
Large exploratory grids and their results do not belong in the pytest inventory.
"""

from contextlib import closing
from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import DriveSource, EnvelopeSpec, ModelSource
from zcu_tools.simulate.fluxonium import (
    DispersiveResonator,
    FluxoniumNoise,
    SingleFluxoniumModel,
)

from .accuracy_cases import AccuracyCase, drive_track, timeline


def real_rwa_source(bias_phi0: float = 0.5) -> ModelSource:
    """Return an owned static 21-level source at bias_phi0 in flux quanta.

    The fixed test inputs are EJ/EC/EL=4/1/1 GHz, native cutoff 40, 50 mK,
    charge coupling .1 GHz/gain and the loss/resonator parameters below.
    The public numeric model supplies Gibbs density, bases, drives and baths.
    All requested levels reuse the DC model, as these sweeps have no flux
    waveform. No environment, device or settings file is used (ADR-0072).
    The caller must close the returned source. Numeric model errors propagate.
    """
    physical = SingleFluxoniumModel(
        (4.0, 1.0, 1.0),
        bias_phi0,
        cutoff=40,
        retained_levels=21,
        temperature_k=0.05,
    )
    charge = physical.charge_operator()
    with closing(
        physical.model_source(
            (DriveSource("q1_charge", 0.1 * charge, -0.0 * charge),),
            noise=FluxoniumNoise(q_cap=1e6, q_ind=1e8, xqp=2e-7, aphi=2e-6),
            resonator=DispersiveResonator(
                frequency_ghz=7.0,
                g_ghz=0.03,
                kappa_per_ns=2 * np.pi * (1.0 + 0.1) / 1000,
                thermal_occupation=0.0,
            ),
        )
    ) as numeric:
        model = numeric.model((0,))
        initial = numeric.initial_density.copy()
    return ModelSource(initial, lambda level: replace(model, level=level))


def real_rwa_case(
    source: ModelSource,
    *,
    detuning_ghz: float,
    duration_ns: float,
    gain: float,
    phase_rad: float = 0.0,
) -> AccuracyCase:
    """Build one constant pulse on a caller-owned static real-model source.

    detuning_ghz is signed carrier minus DC E1-E0. duration_ns is the pulse and
    shot length in ns; gain is dimensionless; phase_rad is absolute phase at
    t=0. There is no idle tail, readout or hardware quantization. The model,
    initial density and bath are unchanged. Only the first model drive is used.
    The caller retains ownership of source, including on construction failure.
    Invalid numeric inputs follow TimelineBatch/Engine validation.
    """
    model = source.model((0,))
    center = float(model.energies_ghz[1] - model.energies_ghz[0])
    batch = timeline(1, duration_ns)
    track = drive_track(
        batch,
        EnvelopeSpec("constant"),
        frequency=center + detuning_ghz,
        gain=gain,
        port=model.drives[0].name,
    )
    track = replace(track, phase_rad=track.phase_rad * 0 + phase_rad)
    return AccuracyCase(
        f"real_rwa/detuning={detuning_ghz:.17g}/duration={duration_ns:.17g}"
        f"/gain={gain:.17g}/phase={phase_rad:.17g}",
        source,
        replace(batch, tracks=(track,)),
        ("single_tone_rwa", "t4"),
    )
