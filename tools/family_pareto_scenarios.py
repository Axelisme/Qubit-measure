"""Tool-owned public-program scans over fixed models, without lab I/O."""

from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import asdict, replace

import numpy as np
from zcu_tools.program import build_soccfg
from zcu_tools.program.v2 import Pulse, Readout, SweepCfg, TimelineBuilder
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg, GaussWaveformCfg
from zcu_tools.program.v2.simulation.compiled import build_compiled_request
from zcu_tools.simulate.evolution import (
    FluxExpansion,
    ModelSource,
    SecularModel,
    segment_spans,
)
from zcu_tools.simulators.ftf.config import parse_ftf
from zcu_tools.simulators.ftf.environment import FTFSource, ftf, plan_template
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment import (
    SingleFluxoniumSource,
    single_fluxonium,
)

from tests.simulate.evolution.contract.accuracy_cases import AccuracyCase
from tools.family_pareto_inputs import Input
from tools.family_pareto_settings import (
    BOARD_PROFILE,
    FTF_MODEL,
    INTEGER_MODEL,
    SINGLE_MODEL,
    SOURCE_COMMIT,
)

_PLATEAU = ("dyson", "single_tone_rwa", "floquet_period", "t6")
_SHAPED = ("single_tone_gate", "shaped_sambe", "gate_cache", "period_table", "dop853")


def _single_input(
    name: str,
    emit: Callable[[TimelineBuilder, float], None],
    manifest: dict[str, object],
    families: tuple[str, ...],
) -> Input:
    """Own single-model resources; compile a core sequence before fixed readout."""
    settings = parse_single_fluxonium(SINGLE_MODEL)
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
        build = _numeric_builder(AccuracyCase(name, numeric, batch, families))

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
    return _declared_input(name, build, manifest)


def _numeric_builder(
    case: AccuracyCase, *, expansion: FluxExpansion | None = None
) -> Callable[[], AccuracyCase]:
    """Detach public numeric inputs while their environment owner is still open."""
    name, batch, families = case.name, case.batch, case.families
    initial = case.source.initial_density
    subdivisions = case.source.slow_envelope_segments_per_period
    models: dict[tuple[int, ...], SecularModel] = {
        (level := tuple(int(value) for value in row)): case.source.model(level)
        for row in np.unique(batch.flux.level, axis=0)
    }

    def build() -> AccuracyCase:
        return AccuracyCase(
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


def _declared_input(
    name: str, build: Callable[[], AccuracyCase], manifest: dict[str, object]
) -> Input:
    """Declare driven/idle logical work from input supports, not Engine telemetry."""
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


def carrier_sweep_input() -> Input:
    """Compile 24 carriers across DC E1-E0 +/-100 MHz, full 21-level model.

    The 1 us constant drive uses uncalibrated gain 0.1. Resources close before
    return; Input.build supplies a fresh caller-owned numeric source. All timing
    and amplitudes come from the public compiler/environment conversion.
    """

    def emit(builder: TimelineBuilder, center: float) -> None:
        carrier = builder.add_sweep(
            "carrier",
            SweepCfg(start=center - 100, stop=center + 100, step=200 / 23, expts=24),
        )
        builder.emit(
            Pulse(
                "drive",
                PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=center,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=1.0),
                ),
                freq=carrier,
            )
        )

    return _single_input(
        "carrier_sweep",
        emit,
        {
            "pulse_sequence": "constant drive, then terminal readout",
            "waveform": "constant",
            "pulse_length_us": 1.0,
            "gain": 0.1,
            "carrier_offset_range_mhz": [-100, 100],
        },
        _PLATEAU,
    )


def gain_sweep_input() -> Input:
    """Compile 24 Gaussian gains .01.. .1 at fixed model E1-E0 (21 levels).

    The 100 ns envelope has sigma 20 ns. These are uncalibrated benchmark
    settings. Ownership, compilation and failure semantics match carrier_sweep_input.
    """

    def emit(builder: TimelineBuilder, center: float) -> None:
        gain = builder.add_sweep(
            "gain", SweepCfg(start=0.01, stop=0.1, step=0.09 / 23, expts=24)
        )
        builder.emit(
            Pulse(
                "drive",
                PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=center,
                    gain=0.1,
                    waveform=GaussWaveformCfg(length=0.1, sigma=0.02),
                ),
                gain=gain,
            )
        )

    return _single_input(
        "gain_sweep",
        emit,
        {
            "pulse_sequence": "Gaussian drive, then terminal readout",
            "waveform": "gauss",
            "pulse_length_us": 0.1,
            "sigma_us": 0.02,
            "gain_range": [0.01, 0.1],
        },
        _SHAPED,
    )


def length_sweep_input() -> Input:
    """Compile 24 constant-pulse lengths .02.. .4 us at model E1-E0.

    Gain is fixed at .1, without a fitted calibration. Resource ownership and
    compiler/environment failures follow carrier_sweep_input.
    """

    def emit(builder: TimelineBuilder, center: float) -> None:
        # Quantize each requested endpoint, rather than accumulating a truncated step.
        length = builder.add_table_sweep("length", np.linspace(0.02, 0.4, 24).tolist())
        builder.emit(
            Pulse(
                "drive",
                PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=center,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=0.02),
                ),
                length=length,
            )
        )

    return _single_input(
        "length_sweep",
        emit,
        {
            "pulse_sequence": "variable-length constant drive, then terminal readout",
            "waveform": "constant",
            "pulse_length_range_us": [0.02, 0.4],
            "gain": 0.1,
        },
        _PLATEAU,
    )


def idle_sweep_input() -> Input:
    """Compile two fixed 20 ns gates separated by 24 idles .02.. .4 us.

    The second gate is Y-phase; neither gate is a calibrated pi/2 rotation.
    Ownership and compilation failures follow carrier_sweep_input.
    """

    def emit(builder: TimelineBuilder, center: float) -> None:
        idle = builder.add_sweep(
            "idle", SweepCfg(start=0.02, stop=0.4, step=0.38 / 23, expts=24)
        )
        first = Pulse(
            "first",
            PulseCfg(
                ch=1,
                nqz=1,
                freq=center,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.02),
            ),
        )
        second = Pulse(
            "second",
            PulseCfg(
                ch=1,
                nqz=1,
                freq=center,
                phase=90,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.02),
            ),
        )
        builder.emit(first, timeline="default")
        builder.timeline_inc(idle)
        builder.emit(second, timeline="default")

    return _single_input(
        "idle_sweep",
        emit,
        {
            "pulse_sequence": "fixed X-phase gate, variable idle, fixed Y-phase gate, terminal readout",
            "waveform": "constant",
            "pulse_length_us": 0.02,
            "gain": 0.1,
            "idle_range_us": [0.02, 0.4],
        },
        _PLATEAU,
    )


def gate_sequence_input() -> Input:
    """Compile 24 deterministic prefixes of a four-phase gate alphabet.

    Prefix lengths are 1..24; each constant pulse lasts 20 ns at gain .1.
    These are benchmark gates, not calibrated rotations or random RB sequences.
    Ownership and compilation failures follow carrier_sweep_input.
    """

    def emit(builder: TimelineBuilder, center: float) -> None:
        builder.add_loop("count", 24)
        gates = [
            Pulse(
                f"gate_{phase}",
                PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=center,
                    phase=phase,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=0.02),
                ),
            )
            for phase in (0, 90, 180, 270)
        ]
        with builder.branch(
            by="count", align_timeline=False, timeline="default"
        ) as cases:
            for point in range(24):
                with cases.case(point):
                    for index in range(point + 1):
                        builder.emit(gates[index % len(gates)])

    return _single_input(
        "gate_sequence",
        emit,
        {
            "pulse_sequence": "1..24 gates from repeated X,Y,-X,-Y phase alphabet, terminal readout",
            "waveform": "constant",
            "pulse_length_us": 0.02,
            "gain": 0.1,
            "gate_counts": list(range(1, 25)),
            "gate_phases_deg": [0, 90, 180, 270],
        },
        _PLATEAU,
    )


def long_gate_sequence_input() -> Input:
    """Compile N's constant-gate prefixes at 50, 100 and 200 gates.

    Keep N's full 21-level model, DC E1-E0 carrier, 20 ns support, gain .1
    and X,Y,-X,-Y phase cycle. Public compilation preserves alignment gaps
    and terminal readout lead. These are uncalibrated benchmark gates.
    Resources close before return; Input.build supplies a fresh caller-owned
    numeric source. Compiler/environment failures propagate as in gate_sequence_input.
    """
    gate_counts = (50, 100, 200)

    def emit(builder: TimelineBuilder, center: float) -> None:
        builder.add_loop("count", len(gate_counts))
        gates = [
            Pulse(
                f"gate_{phase}",
                PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=center,
                    phase=phase,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=0.02),
                ),
            )
            for phase in (0, 90, 180, 270)
        ]
        with builder.branch(
            by="count", align_timeline=False, timeline="default"
        ) as cases:
            for point, count in enumerate(gate_counts):
                with cases.case(point):
                    for index in range(count):
                        builder.emit(gates[index % len(gates)])

    return _single_input(
        "long_gate_sequence",
        emit,
        {
            "pulse_sequence": "50/100/200 gates from repeated X,Y,-X,-Y phase alphabet, terminal readout",
            "waveform": "constant",
            "pulse_length_us": 0.02,
            "gain": 0.1,
            "gate_counts": list(gate_counts),
            "gate_phases_deg": [0, 90, 180, 270],
        },
        _PLATEAU,
    )


def shaped_gate_sequence_input() -> Input:
    """Compile 24 Gaussian prefixes of the X,Y,-X,-Y gate alphabet.

    Gate counts are 1..24 on the fixed 21-level model at DC E1-E0. Each gate
    lasts 100 ns with sigma 20 ns and uncalibrated gain .1. These durations
    match the shaped gain scan, not a calibrated rotation. Resources close
    before return; Input.build supplies a fresh caller-owned numeric source.
    Compiler/environment failures propagate as in carrier_sweep_input.
    """

    def emit(builder: TimelineBuilder, center: float) -> None:
        builder.add_loop("count", 24)
        gates = [
            Pulse(
                f"gate_{phase}",
                PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=center,
                    phase=phase,
                    gain=0.1,
                    waveform=GaussWaveformCfg(length=0.1, sigma=0.02),
                ),
            )
            for phase in (0, 90, 180, 270)
        ]
        with builder.branch(
            by="count", align_timeline=False, timeline="default"
        ) as cases:
            for point in range(24):
                with cases.case(point):
                    for index in range(point + 1):
                        builder.emit(gates[index % len(gates)])

    return _single_input(
        "shaped_gate_sequence",
        emit,
        {
            "pulse_sequence": "1..24 Gaussian gates from repeated X,Y,-X,-Y phase alphabet, terminal readout",
            "waveform": "gauss",
            "pulse_length_us": 0.1,
            "sigma_us": 0.02,
            "gain": 0.1,
            "gate_counts": list(range(1, 25)),
            "gate_phases_deg": [0, 90, 180, 270],
        },
        _SHAPED,
    )


def long_shaped_gate_sequence_input() -> Input:
    """Compile S's Gaussian-gate prefixes at 50, 100 and 200 gates.

    Keep S's full 21-level model, DC E1-E0 carrier, 100 ns support, 20 ns
    sigma, gain .1 and X,Y,-X,-Y phase cycle. Public compilation preserves
    alignment gaps and terminal readout lead. These are uncalibrated gates.
    Resources close before return; Input.build supplies a fresh caller-owned
    numeric source. Failures propagate as in shaped_gate_sequence_input.
    """
    gate_counts = (50, 100, 200)

    def emit(builder: TimelineBuilder, center: float) -> None:
        builder.add_loop("count", len(gate_counts))
        gates = [
            Pulse(
                f"gate_{phase}",
                PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=center,
                    phase=phase,
                    gain=0.1,
                    waveform=GaussWaveformCfg(length=0.1, sigma=0.02),
                ),
            )
            for phase in (0, 90, 180, 270)
        ]
        with builder.branch(
            by="count", align_timeline=False, timeline="default"
        ) as cases:
            for point, count in enumerate(gate_counts):
                with cases.case(point):
                    for index in range(count):
                        builder.emit(gates[index % len(gates)])

    return _single_input(
        "long_shaped_gate_sequence",
        emit,
        {
            "pulse_sequence": "50/100/200 Gaussian gates from repeated X,Y,-X,-Y phase alphabet, terminal readout",
            "waveform": "gauss",
            "pulse_length_us": 0.1,
            "sigma_us": 0.02,
            "gain": 0.1,
            "gate_counts": list(gate_counts),
            "gate_phases_deg": [0, 90, 180, 270],
        },
        _SHAPED,
    )


def simultaneous_drive_input() -> Input:
    """Compile a common gain scan over synchronous F1/F2 Gaussian drives.

    The tool retains 32 joint states with local 4x4x4. Only retained_levels
    overrides the fixed source model. At the specified CZ working point at
    least 25 joint states were needed for conditional phase within 0.5 degrees;
    32 leaves margin, not proof for arbitrary CZ or full native/local convergence.
    Carriers use uncoupled local DC E1-E0, not fitted joint gate calibrations.
    """
    settings = parse_ftf(FTF_MODEL)
    settings = settings.model_copy(
        update={
            "drift": None,
            "numerics": settings.numerics.model_copy(update={"time_scale": 0.0}),
        }
    )
    board = build_soccfg(BOARD_PROFILE)
    env = ftf(settings, soccfg=board)
    families = ("common_frame", "multitone_gate", "multitone_rwa", "floquet", "dop853")
    with ExitStack() as stack:
        stack.callback(env.devices.close_all_devices)
        if env.close_backend is not None:
            stack.callback(env.close_backend)
        owner = stack.enter_context(
            closing(FTFSource(settings, env.devices, soccfg=board))
        )
        snapshot = owner.snapshot()
        facts = owner.model_facts(snapshot)
        numeric = stack.enter_context(closing(facts.source))
        centers = [
            float((energies[1] - energies[0]) * 1000)
            for energies in (
                facts.joint.local_energies_ghz[0],
                facts.joint.local_energies_ghz[2],
            )
        ]
        builder = TimelineBuilder(board, reps=1, relax_delay=1.0)
        gain = builder.add_sweep(
            "gain", SweepCfg(start=0.01, stop=0.1, step=0.09 / 23, expts=24)
        )
        builder.fork("default", into="second")
        for channel, center, timeline in zip(
            (2, 3), centers, ("default", "second"), strict=True
        ):
            builder.emit(
                Pulse(
                    f"drive_{channel}",
                    PulseCfg(
                        ch=channel,
                        nqz=1,
                        freq=center,
                        gain=0.1,
                        waveform=GaussWaveformCfg(length=0.1, sigma=0.02),
                    ),
                    gain=gain,
                ),
                timeline=timeline,
            )
        builder.join("second", into="default")
        builder.emit(
            Readout(
                "readout",
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=settings.readout.resonators.r1.frequency_ghz * 1000,
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
        batch = plan_template(
            settings,
            snapshot,
            template,
            time_base=request.time_base,
            soccfg=board,
            facts=facts,
        ).batch
        build = _numeric_builder(
            AccuracyCase("simultaneous_drive", numeric, batch, families),
            expansion=numeric.flux_expansion(),
        )
    return _declared_input(
        "simultaneous_drive",
        build,
        {
            "pulse_sequence": "synchronous F1/F2 Gaussian drives, then terminal readout",
            "waveform": "gauss",
            "pulse_length_us": 0.1,
            "sigma_us": 0.02,
            "gain_range": [0.01, 0.1],
            "source_commit": SOURCE_COMMIT,
            "model_source": "fixed FTF physical parameters; retained_levels overridden from 16 to 32",
            "retained_levels_reason": "specified CZ working point with local 4x4x4 needed at least 25 joint states for conditional phase within 0.5 degrees; 32 leaves margin, not arbitrary CZ or full native/local convergence",
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
                "centers_mhz": centers,
                "center_method": "uncoupled local DC E1-E0 for F1 and F2",
                "gain_method": "fixed uncalibrated benchmark gains shared by both ports",
                "readout_method": "fixed bare r1 cavity frequency and flight time; gain 0.1 and 1 us pulse/window",
            },
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
        build = _numeric_builder(AccuracyCase(name, numeric, batch, families))
    return _declared_input(
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


def overlapping_shaped_tones_input() -> Input:
    """Compile 24 Stokes-first Gaussian leads from disjoint to full overlap.

    Each port addresses the same integer-fluxonium charge operator. Requested
    support/sigma are 0.3/0.05 us. Carriers and converted gains start from the
    historical synthetic STIRAP controls, not a calibrated transfer claim.
    No inventory repetitions are added for multitone_gate's cost preference.
    """
    shaped = ("single_tone_gate", "shaped_sambe", "gate_cache", "period_table")
    overlapping = ("common_frame", "multitone_gate", "multitone_rwa")
    single_companion = "common_frame"
    overlap_companion = "shaped_sambe"

    def emit(builder: TimelineBuilder, _model: SecularModel) -> None:
        lead = builder.add_table_sweep("lead", np.linspace(0.3, 0.0, 24).tolist())
        builder.fork("default", into="pump")
        builder.emit(
            Pulse(
                "stokes",
                PulseCfg(
                    ch=2,
                    nqz=1,
                    freq=3111.1686681315896,
                    gain=0.100006103888177,
                    waveform=GaussWaveformCfg(length=0.3, sigma=0.05),
                ),
            )
        )
        builder.emit(
            Pulse(
                "pump",
                PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=7241.848921606473,
                    gain=0.030001831166453,
                    pre_delay=lead,
                    waveform=GaussWaveformCfg(length=0.3, sigma=0.05),
                ),
            ),
            timeline="pump",
        )
        builder.join("pump", into="default")

    item = integer_fluxonium_input(
        "overlapping_shaped_tones",
        emit,
        {
            "pulse_sequence": "Stokes-first Gaussian pair with decreasing lead, then terminal readout",
            "waveform": "gauss",
            "pulse_length_us": 0.3,
            "sigma_us": 0.05,
            "lead_range_us": [0.3, 0.0],
            "carriers_mhz": {"Pump": 7241.848921606473, "Stokes": 3111.1686681315896},
            "gains": {"Pump": 0.030001831166453, "Stokes": 0.100006103888177},
            "controls_reason": "historical synthetic STIRAP carriers and envelope-scale-converted gains; current delay-form width/support scale; not calibrated transfer",
            "inventory_reason": "one pair per delay, no repetitions to satisfy multitone_gate cost",
        },
        (*shaped, *overlapping, "dop853"),
    )
    counts = [0, 0, 0]
    durations = [0.0, 0.0, 0.0]
    with closing((case := item.build()).source):
        for point, spans in enumerate(segment_spans(case.batch)):
            for start, end in spans:
                tones = sum(
                    bool(
                        track.active[point]
                        and track.start_ns[point] <= start
                        and end <= track.start_ns[point] + track.duration_ns[point]
                    )
                    for track in case.batch.tracks
                )
                counts[tones] += 1
                durations[tones] += end - start
    idle, single, multiple = [(counts[i], durations[i]) for i in range(3)]
    expected = {
        family: {family: single, single_companion: multiple, "t4": idle}
        for family in shaped
    }
    expected.update(
        {
            family: {family: multiple, overlap_companion: single, "t4": idle}
            for family in overlapping
        }
    )
    expected["dop853"] = {
        "dop853": (counts[1] + counts[2], durations[1] + durations[2]),
        "t4": idle,
    }
    companions = {
        **{family: (single_companion, "t4") for family in shaped},
        **{family: (overlap_companion, "t4") for family in overlapping},
        "dop853": ("t4",),
    }
    # This named group may share candidates only while the full route
    # declaration proves that the target retains every single-tone span.
    expected["period_table"] = {"period_table": single, "dop853": multiple, "t4": idle}
    companions["period_table"] = ("dop853", "t4")
    return replace(
        item,
        expected_routes=expected,
        companions_by_target=companions,
        manifest={
            **item.manifest,
            "expected_routes": expected,
            "support_work": {"idle": idle, "single": single, "overlap": multiple},
        },
    )


def overlapping_constant_tones_input() -> Input:
    """Compile a fixed carrier and 24 scanned carriers with full constant overlap.

    Both charge lines act on the same full integer-fluxonium model. Pump stays
    at DC E1-E0; Stokes scans +/-50 MHz around it. Both supports are 1 us.
    Gains .02/.002 preserve the generic reset's physical charge-drive scale
    on the integer model's stronger line, with a tenfold weaker probe. These
    controls are benchmark settings, not a calibrated reset or AC Stark scan.
    Input.build returns a fresh caller-owned source that must be closed.
    """
    carriers: dict[str, object] = {}

    def emit(builder: TimelineBuilder, model: SecularModel) -> None:
        center = float((model.energies_ghz[1] - model.energies_ghz[0]) * 1000)
        carriers.update(
            {
                "fixed_carrier_mhz": center,
                "scanned_carrier_range_mhz": [center - 50, center + 50],
            }
        )
        carrier = builder.add_sweep(
            "carrier",
            SweepCfg(start=center - 50, stop=center + 50, step=100 / 23, expts=24),
        )
        builder.fork("default", into="probe")
        builder.emit(
            Pulse(
                "pump",
                PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=center,
                    gain=0.02,
                    waveform=ConstWaveformCfg(length=1.0),
                ),
            )
        )
        builder.emit(
            Pulse(
                "probe",
                PulseCfg(
                    ch=2,
                    nqz=1,
                    freq=center,
                    gain=0.002,
                    waveform=ConstWaveformCfg(length=1.0),
                ),
                freq=carrier,
            ),
            timeline="probe",
        )
        builder.join("probe", into="default")

    item = integer_fluxonium_input(
        "overlapping_constant_tones",
        emit,
        {
            "pulse_sequence": "equal-length fully overlapping constant charge tones, fixed Pump and scanned Stokes, then terminal readout",
            "waveform": "constant",
            "pulse_length_us": 1.0,
            "gains": {"Pump": 0.02, "Stokes": 0.002},
            "carrier_method": "DC model E1-E0, Stokes offset +/-50 MHz",
            "controls_reason": "dual-tone reset core and +/-50 MHz form range; generic reset 1 us support and physical drive scale, tenfold weaker probe; not calibrated",
        },
        ("floquet", "common_frame", "multitone_rwa", "dop853"),
    )
    return replace(item, manifest={**item.manifest, **carriers})


def make_real_inputs(scenario: str | None = None) -> list[Input]:
    """Build all fixed scenarios, or only the named Input before any model work.

    Returned inputs own detached fresh-source factories. An unknown scenario
    raises ValueError listing legal Input names without constructing models.
    """
    builders = {
        "carrier_sweep": carrier_sweep_input,
        "gain_sweep": gain_sweep_input,
        "length_sweep": length_sweep_input,
        "idle_sweep": idle_sweep_input,
        "gate_sequence": gate_sequence_input,
        "simultaneous_drive": simultaneous_drive_input,
        "shaped_gate_sequence": shaped_gate_sequence_input,
        "overlapping_shaped_tones": overlapping_shaped_tones_input,
        "overlapping_constant_tones": overlapping_constant_tones_input,
        "long_gate_sequence": long_gate_sequence_input,
        "long_shaped_gate_sequence": long_shaped_gate_sequence_input,
    }
    if scenario is None:
        return [build() for build in builders.values()]
    if scenario not in builders:
        raise ValueError(
            f"Unknown scenario: {scenario}. Legal names: {', '.join(builders)}"
        )
    return [builders[scenario]()]
