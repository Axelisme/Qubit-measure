"""Fixed-model compilation, numeric detachment and route declarations.

Scene controls live in family_pareto_scenarios. This owner contains the shared
single/integer model lifecycle used by multiple independent scan declarations.
"""

from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import asdict

import numpy as np
from zcu_tools.program import build_soccfg
from zcu_tools.program.v2 import Readout, TimelineBuilder
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation.compiled import build_compiled_request
from zcu_tools.simulate.evolution import (
    FluxExpansion,
    ModelSource,
    SecularModel,
    segment_spans,
)
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment import (
    SingleFluxoniumSource,
    single_fluxonium,
)

from tools.family_pareto_inputs import Input, Scene
from tools.family_pareto_settings import (
    BOARD_PROFILE,
    INTEGER_MODEL,
    SINGLE_MODEL,
    SOURCE_COMMIT,
)


def single_model_input(
    name: str,
    emit: Callable[[TimelineBuilder, float], None],
    manifest: dict[str, object],
    families: tuple[str, ...],
    *,
    dc_bias_phi0: float | None = None,
) -> Input:
    """Compile a caller's pulse sequence on the fixed 21-level single model.

    name identifies the returned real Input and each built Scene.
    emit(builder, center) runs once with a public TimelineBuilder and the DC
    E1-E0 carrier center in MHz. Use public Pulse/TimelineBuilder operations
    to emit the complete sequence or scan before terminal readout; the charge
    drive uses generator 1. Pulse frequencies are MHz and durations are us.
    This builder then appends generator 0/ADC 0 readout and compiles the input.
    dc_bias_phi0 optionally replaces the fixed DC offset (.5 Phi0), without
    changing the live DC source value. Drift is disabled and time_scale is zero.

    manifest supplies caller provenance: pulse_sequence is required prose;
    other JSON-compatible entries can describe axes, gains, carriers and units.
    Generated model/settings, calibration, timing and route facts take precedence
    over caller entries with the same keys. families is a tuple of evolution
    family names, e.g. ("dyson", "single_tone_rwa"), to declare pinned targets
    for driven spans, with t4 as the idle companion; it does not bypass guards.

    Construction closes the environment, device and temporary source resources
    before returning. Each Input.build() creates an Scene with a fresh
    detached numeric source; the caller must close that case's source.
    Callback, model, compilation and cleanup errors propagate. A missing
    pulse_sequence raises KeyError; provenance serialization errors propagate
    when the manifest is frozen.
    """
    settings = parse_single_fluxonium(SINGLE_MODEL)
    if dc_bias_phi0 is not None:
        settings = settings.model_copy(
            update={
                "dc_bias": settings.dc_bias.model_copy(
                    update={"offset_phi0": dc_bias_phi0}
                )
            }
        )
    settings = settings.model_copy(
        update={
            "drift": None,
            "numerics": settings.numerics.model_copy(update={"time_scale": 0.0}),
        }
    )
    board = build_soccfg(BOARD_PROFILE)
    env = single_fluxonium(settings, soccfg=board)
    with ExitStack() as stack:
        stack.callback(env.devices.close_all_devices)
        if env.close_backend is not None:
            stack.callback(env.close_backend)
        owner = stack.enter_context(
            closing(SingleFluxoniumSource(settings, env.devices))
        )
        snapshot = owner.snapshot()
        numeric = stack.enter_context(closing(owner.model_facts(snapshot).source))
        model = numeric.model((0,))
        center_mhz = float((model.energies_ghz[1] - model.energies_ghz[0]) * 1000)
        builder = TimelineBuilder(board, reps=1, relax_delay=1.0)
        emit(builder, center_mhz)
        builder.emit(
            Readout(
                "readout",
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=settings.readout.resonator.frequency_ghz * 1000,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=1.0),
                    ro_length=1.0,
                    trig_offset=settings.readout.adc.time_of_flight_ns / 1000,
                ),
            )
        )
        request = build_compiled_request(
            builder.build(),
            "integrated",
            env.soc.binding,
            0,
            routes=env.soc.routes,
        )
        (template,) = request.templates
        batch = owner.plan_template(
            snapshot, template, time_base=request.time_base
        ).batch
        build = numeric_builder(Scene(name, numeric, batch, families))

    manifest = {
        **manifest,
        "source_commit": SOURCE_COMMIT,
        "model_source": "fixed single-fluxonium physical parameters",
        "effective_settings": settings.model_dump(mode="json"),
        "board_configuration": BOARD_PROFILE,
        "snapshot": asdict(snapshot),
        "time_base": {
            "ticks_per_us_numerator": request.time_base.ticks_per_us.numerator,
            "ticks_per_us_denominator": request.time_base.ticks_per_us.denominator,
        },
        "compiled_carriers_ghz": [
            track.frequency_ghz.tolist() for track in batch.tracks
        ],
        "calibration": {
            "center_mhz": center_mhz,
            "center_method": "DC model E1-E0",
            "gain_method": "fixed uncalibrated benchmark gains",
            "readout_method": "fixed bare cavity frequency and flight time; gain 0.1 and 1 us pulse/window",
        },
    }
    return declared_input(name, build, manifest)


def numeric_builder(
    case: Scene, *, expansion: FluxExpansion | None = None
) -> Callable[[], Scene]:
    """Return a fresh-source factory detached from an open environment's Scene.

    Snapshot the scene's initial density, scheduled models, batch, family names
    and slow-envelope policy. expansion optionally supplies a continuous-flux
    Hamiltonian. Call before closing the environment-owned source. Each factory
    call returns a new caller-owned source/history; close it after use. Only
    scheduled levels can be requested. Model errors propagate during snapshot.
    """
    name, batch, families = case.name, case.batch, case.families
    initial = case.source.initial_density
    subdivisions = case.source.slow_envelope_segments_per_period
    models: dict[tuple[int, ...], SecularModel] = {
        (level := tuple(int(value) for value in row)): case.source.model(level)
        for row in np.unique(batch.flux.level, axis=0)
    }

    def build() -> Scene:
        return Scene(
            name,
            ModelSource(
                initial,
                models.__getitem__,
                expansion,
                slow_envelope_segments_per_period=subdivisions,
            ),
            batch,
            families,
        )

    return build


def declared_input(
    name: str, build: Callable[[], Scene], manifest: dict[str, object]
) -> Input:
    """Declare real-scene targets/idle companions from public timeline supports.

    name is the stable inventory label. build returns a fresh owned Scene and
    is called once here, closing its source after inspecting segment_spans.
    manifest must contain pulse_sequence prose; the returned Input adds point,
    span and target-route facts. Each declared family owns every driven span,
    with t4 as idle companion. Missing prose raises KeyError; builder and
    numeric-support failures propagate. No dispatch or density evolution runs.
    """
    with closing((case := build()).source):
        batch = case.batch
        driven_count = idle_count = 0
        driven_ns = idle_ns = 0.0
        spans_per_point = []
        for point, spans in enumerate(segment_spans(batch)):
            spans_per_point.append(len(spans))
            for start, end in spans:
                playing = any(
                    track.active[point]
                    and track.start_ns[point] <= start
                    and end <= track.start_ns[point] + track.duration_ns[point]
                    for track in batch.tracks
                )
                if playing:
                    driven_count += 1
                    driven_ns += end - start
                else:
                    idle_count += 1
                    idle_ns += end - start
        expected = {
            family: {family: (driven_count, driven_ns), "t4": (idle_count, idle_ns)}
            for family in case.families
        }
    return Input(
        name,
        build,
        str(manifest["pulse_sequence"]),
        category="real",
        companions=("t4",),
        expected_routes=expected,
        metadata={"sweep_type": name, "companions": ("t4",)},
        manifest={
            **manifest,
            "points": len(batch.points),
            "spans": driven_count + idle_count,
            "spans_per_point": spans_per_point,
            "expected_routes": expected,
        },
    )


def integer_fluxonium_input(
    name: str,
    emit: Callable[[TimelineBuilder, SecularModel], None],
    manifest: dict[str, object],
    families: tuple[str, ...],
) -> Input:
    """Compile a core sequence on the fixed 80-native/20-retained integer model.

    emit receives a public TimelineBuilder and the DC SecularModel (GHz), and
    emits the entire scan before readout. Pump uses gen1 and Stokes gen2; both
    couple to the same charge operator at 1 GHz/gain. Constant or sampled
    waveforms and public sweeps are supported. No drive is added by this builder.
    manifest must include pulse_sequence prose and should describe requested
    axes, gains, carriers and units. families lists admissible pinned targets.
    The returned Input owns a fresh detached numeric source on each build;
    callers close it. It declares each target for all driven spans plus t4 for
    idle; mixed-domain callers replace expected_routes and companion declarations.
    Compilation, model, resource cleanup and malformed manifest errors propagate.
    """
    settings = parse_single_fluxonium(INTEGER_MODEL)
    board = build_soccfg(BOARD_PROFILE)
    env = single_fluxonium(settings, soccfg=board)
    with ExitStack() as stack:
        stack.callback(env.devices.close_all_devices)
        if env.close_backend is not None:
            stack.callback(env.close_backend)
        owner = stack.enter_context(
            closing(SingleFluxoniumSource(settings, env.devices))
        )
        snapshot = owner.snapshot()
        numeric = stack.enter_context(closing(owner.model_facts(snapshot).source))
        builder = TimelineBuilder(board, reps=1, relax_delay=1.0)
        emit(builder, numeric.model((0,)))
        builder.emit(
            Readout(
                "readout",
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=settings.readout.resonator.frequency_ghz * 1000,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=1.0),
                    ro_length=1.0,
                    trig_offset=settings.readout.adc.time_of_flight_ns / 1000,
                ),
            )
        )
        request = build_compiled_request(
            builder.build(), "integrated", env.soc.binding, 0, routes=env.soc.routes
        )
        (template,) = request.templates
        batch = owner.plan_template(
            snapshot, template, time_base=request.time_base
        ).batch
        build = numeric_builder(Scene(name, numeric, batch, families))
    return declared_input(
        name,
        build,
        {
            **manifest,
            "source_commit": SOURCE_COMMIT,
            "model_source": "fixed integer-fluxonium physical parameters, native 80/retained 20",
            "effective_settings": settings.model_dump(mode="json"),
            "board_configuration": BOARD_PROFILE,
            "snapshot": asdict(snapshot),
            "time_base": {
                "ticks_per_us_numerator": request.time_base.ticks_per_us.numerator,
                "ticks_per_us_denominator": request.time_base.ticks_per_us.denominator,
            },
            "compiled_carriers_ghz": [
                track.frequency_ghz.tolist() for track in batch.tracks
            ],
            "calibration": {
                "gain_method": "uncalibrated benchmark gains, not acquired-IQ validation",
                "readout_method": "bare cavity, gain 0.1, 1 us pulse/window, zero flight time",
            },
        },
    )
