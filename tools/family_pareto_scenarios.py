"""Curated public-program scenarios using tool-owned fixed model inputs.

Only core pulse sequences are reproduced. No experiment run, acquisition,
replay, lab import or lab-file read participates in construction.
"""

from contextlib import ExitStack, closing
from dataclasses import asdict

from zcu_tools.program import build_soccfg
from zcu_tools.program.v2 import Pulse, Readout, SweepCfg, TimelineBuilder
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation.compiled import build_compiled_request
from zcu_tools.simulate.evolution import ModelSource, SecularModel, segment_spans
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment import (
    SingleFluxoniumSource,
    single_fluxonium,
)

from tests.simulate.evolution.contract.accuracy_cases import AccuracyCase
from tools.family_pareto_inputs import Input
from tools.family_pareto_settings import BOARD_PROFILE, SINGLE_MODEL, SOURCE_COMMIT

_CARRIER_FAMILIES = ("dyson", "single_tone_rwa", "floquet_period", "t6")


def carrier_sweep_input() -> Input:
    """Compile the fixed 24-carrier, 21-level constant-pulse benchmark.

    The center is the DC model's E1-E0 transition. The fixed range is +/-100
    MHz, the pulse lasts 1 us at gain 0.1, and readout uses the fixed bare
    cavity frequency and flight time. These are illustrative settings, not a
    calibration. Public compiled request/model_facts/plan_template determine
    quantized carriers and the complete pre-readout timeline. Construction
    owns and closes all backend/device/source resources. The returned build
    factory supplies caller-owned fresh numeric sources over immutable models.
    """
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
        initial = numeric.initial_density
        subdivisions = numeric.slow_envelope_segments_per_period
        center_mhz = float((model.energies_ghz[1] - model.energies_ghz[0]) * 1000)

        builder = TimelineBuilder(board, reps=1, relax_delay=1.0)
        carrier = builder.add_sweep(
            "carrier",
            SweepCfg(
                start=center_mhz - 100,
                stop=center_mhz + 100,
                step=200 / 23,
                expts=24,
            ),
        )
        builder.emit(
            Pulse(
                "drive",
                PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=center_mhz,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=1.0),
                ),
                freq=carrier,
            )
        )
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

    models: dict[tuple[int, ...], SecularModel] = {(0,): model}

    def build() -> AccuracyCase:
        return AccuracyCase(
            "carrier_sweep",
            ModelSource(
                initial,
                models.__getitem__,
                slow_envelope_segments_per_period=subdivisions,
            ),
            batch,
            _CARRIER_FAMILIES,
        )

    # Input support, not execution telemetry, supplies the expected partition.
    driven_count = idle_count = 0
    driven_ns = idle_ns = 0.0
    for point, spans in enumerate(segment_spans(batch)):
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
        for family in _CARRIER_FAMILIES
    }
    return Input(
        "carrier_sweep",
        build,
        "Curated carrier scan with constant pulse and idle tail",
        category="real",
        companions=("t4",),
        expected_routes=expected,
        metadata={"sweep_type": "carrier", "companions": ("t4",)},
        manifest={
            "source_commit": SOURCE_COMMIT,
            "model_source": "fixed single-fluxonium physical parameters",
            "effective_settings": settings.model_dump(mode="json"),
            "board_configuration": BOARD_PROFILE,
            "snapshot": asdict(snapshot),
            "time_base": {
                "ticks_per_us_numerator": request.time_base.ticks_per_us.numerator,
                "ticks_per_us_denominator": request.time_base.ticks_per_us.denominator,
            },
            "points": len(batch.points),
            "spans": driven_count + idle_count,
            "expected_routes": expected,
            "pulse_sequence": "constant drive, then terminal readout",
            "waveform": "constant",
            "pulse_length_us": 1.0,
            "gain": 0.1,
            "requested_carrier_range_mhz": [center_mhz - 100, center_mhz + 100],
            "compiled_carriers_ghz": [
                track.frequency_ghz.tolist() for track in batch.tracks
            ],
            "calibration": {
                "center_mhz": center_mhz,
                "center_method": "DC model E1-E0",
                "range_method": "fixed +/-100 MHz",
                "gain_method": "fixed uncalibrated gain 0.1",
                "readout_method": "fixed bare cavity frequency and flight time; gain 0.1 and 1 us pulse/window",
            },
        },
    )


def make_real_inputs() -> list[Input]:
    """Build fixed real-model scenarios with detached fresh-source factories."""
    return [carrier_sweep_input()]
