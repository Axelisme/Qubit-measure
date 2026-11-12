"""Physical inputs for finite constant-RWA sweeps, using the Pareto model owner.

Only input construction lives here. Consumers observe the public diagnostic and
Engine paths, and use interaction_reference for the full-density comparison.
Large exploratory grids and their results do not belong in the pytest inventory.
"""

from contextlib import ExitStack, closing
from dataclasses import replace

from zcu_tools.program import build_soccfg
from zcu_tools.simulate.evolution import EnvelopeSpec, ModelSource
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment import (
    SingleFluxoniumSource,
    single_fluxonium,
)

from tools.family_pareto_settings import BOARD_PROFILE, SINGLE_MODEL

from .accuracy_cases import AccuracyCase, drive_track, timeline


def real_rwa_source(bias_phi0: float = 0.5) -> ModelSource:
    """Return an owned, detached 21-level Pareto source at a fixed DC bias.

    bias_phi0 is DC flux in flux quanta. Only the fixed settings' DC offset is
    changed; native cutoff, retained levels, temperature, bath and drive line
    remain those of SINGLE_MODEL. The public environment owner builds the model
    and initial density. No lab files, device hardware or evolution are used.
    Fake devices and the environment close before return. The caller must close
    the returned numeric source. Configuration/model errors propagate.
    """
    settings = parse_single_fluxonium(SINGLE_MODEL)
    settings = settings.model_copy(
        update={
            "dc_bias": settings.dc_bias.model_copy(update={"offset_phi0": bias_phi0}),
            "drift": None,
            "numerics": settings.numerics.model_copy(update={"time_scale": 0.0}),
        }
    )
    env = single_fluxonium(settings, soccfg=build_soccfg(BOARD_PROFILE))
    with ExitStack() as stack:
        stack.callback(env.devices.close_all_devices)
        if env.close_backend is not None:
            stack.callback(env.close_backend)
        owner = stack.enter_context(
            closing(SingleFluxoniumSource(settings, env.devices))
        )
        numeric = owner.model_facts(owner.snapshot()).source
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
