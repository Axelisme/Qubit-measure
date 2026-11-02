"""Time lookup sweeps preserve scalar scheduling at each compiled scan point."""

from dataclasses import replace

import numpy as np
import pytest
from qick.asm_v2 import QickParam, QickSweep1D
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import SweepCfg, TableParam, TimelineBuilder, ir
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.delay import DelayAuto
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._profiles import load_soccfg
from tests.program.v2.simulation._compiled_support import request


class RegisterDelay(Module):
    name = "register_delay"

    def __init__(self, value):
        self.value = value

    def declare(self, prog):
        parameter = (
            self.value if isinstance(self.value, QickParam) else QickParam(self.value)
        )
        prog.add_reg("hold", init=prog.us2cycles(parameter))

    def lower(self, prog, *, timeline, tag):
        prog.delay_reg_auto("hold")


class PulseAtTime(Module):
    def __init__(self, value, timeline):
        self.name = "timed_drive"
        self.value = value
        self.timeline = timeline
        self.pulse = PulseCfg(
            ch=1, nqz=1, freq=80, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
        ).build("drive")

    def declare(self, prog):
        self.pulse.declare(prog)

    def lower(self, prog, *, timeline, tag):
        prog.pulse(
            1,
            self.pulse.pulse_name(prog),
            t=self.value,
            timeline=self.timeline,
            tag="time",
        )


def time_program(value, *, table=False, linear=False, outer=False, consumer="delay"):
    env = white_noise(load_soccfg(2, 1), std_adc=0, routes=((0, 0),), seed=None)
    builder = TimelineBuilder(env.soccfg, reps=2, initial_delay=0, relax_delay=0.1)
    if outer:
        builder.add_loop("outer", 2)
    if table:
        value = builder.add_table_sweep("point", value)
    elif linear:
        value = builder.add_sweep(
            "point",
            SweepCfg(
                start=value[0],
                stop=value[-1],
                expts=len(value),
                step=(value[-1] - value[0]) / (len(value) - 1),
            ),
        )
    if consumer == "timeline_inc":
        builder.timeline_inc(value, tag="time")
    elif consumer == "delay":
        builder.delay(value, tag="time")
    elif consumer == "register":
        builder.emit(RegisterDelay(value))
    receiver = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
        pre_delay=value if consumer == "pre_delay" else 0.0,
        post_delay=value if consumer == "post_delay" else 0.0,
        trig_offset=value if consumer == "trig_offset" else 0.0,
    ).build("readout")
    if consumer in ("pulse_t", "native_t"):
        builder.emit(
            PulseAtTime(value, "default" if consumer == "pulse_t" else None), tag="time"
        )
    builder.emit(receiver)
    if consumer == "delay_auto":
        builder.delay_auto(value, tag="time")
        builder.emit(receiver)
    elif consumer == "DelayAuto":
        builder.emit(DelayAuto("wait", t=value), tag="time")
        builder.emit(receiver)
    builder.delay_by_timeline()
    return builder.build(), env


def scheduled_shots(prog):
    analysis = ir.analyze_timing(
        ir.CompiledTimingInput(
            prog.prog_list,
            prog.labels,
            prog.timing_layout,
            dmem=prog.compile_datamem(),
            wmem=ir.compiled_wmem(prog),
        )
    )
    return [
        tuple((e.kind, e.cycle - shot.start_cycle, e.wave_words) for e in shot.events)
        for shot in analysis.shots
    ]


@pytest.mark.parametrize(
    "consumer",
    [
        "delay",
        "delay_auto",
        "DelayAuto",
        "timeline_inc",
        "pre_delay",
        "post_delay",
        "trig_offset",
        "pulse_t",
        "native_t",
        "register",
    ],
)
@pytest.mark.parametrize("outer", [False, True])
def test_time_table_matches_scalar_at_every_point_and_rep(consumer, outer):
    values = [0.0213, 0.0107, 0.0354]
    prog, env = time_program(values, table=True, outer=outer, consumer=consumer)
    observed = request(prog, env)
    expected_shots = []
    for point, value in enumerate(values):
        scalar, scalar_env = time_program(value, consumer=consumer)
        assert replace(observed.materialize_point(point), coordinates=()) == request(
            scalar, scalar_env
        ).materialize_point(0)
        expected_shots.append(scheduled_shots(scalar)[0])
    assert scheduled_shots(prog) == expected_shots * (4 if outer else 2)


@pytest.mark.parametrize(
    "consumer",
    [
        "delay",
        "delay_auto",
        "DelayAuto",
        "timeline_inc",
        "pre_delay",
        "post_delay",
        "trig_offset",
        "pulse_t",
        "native_t",
    ],
)
def test_time_tables_match_reencodable_linear_actual_points(consumer):
    values = [0.015625, 0.03125, 0.046875]
    linear, linear_env = time_program(values, linear=True, consumer=consumer)
    parameter = QickSweep1D("point", values[0], values[-1])
    linear.us2cycles(parameter)
    actual = (
        parameter.get_rounded({"point": len(values)})
        .to_array({"point": len(values)})
        .tolist()
    )
    scalar_shots = []
    for point, value in enumerate(actual):
        scalar, scalar_env = time_program(value, consumer=consumer)
        scalar_shots.append(scheduled_shots(scalar)[0])
        assert replace(
            request(linear, linear_env).materialize_point(point), coordinates=()
        ) == request(scalar, scalar_env).materialize_point(0)
    # Prerequisite: each actual physical point must compile back to the exact
    # linear output words and cycles before it serves as a table parity oracle.
    assert scheduled_shots(linear) == scalar_shots * 2
    table, table_env = time_program(actual, table=True, consumer=consumer)
    assert scheduled_shots(table) == scheduled_shots(linear)
    for point in range(len(actual)):
        assert request(table, table_env).materialize_point(point) == request(
            linear, linear_env
        ).materialize_point(point)


@pytest.mark.parametrize("consumer", ["delay", "timeline_inc", "native_t", "pulse_t"])
@pytest.mark.parametrize("constant", [False, True])
def test_table_time_getter_returns_exact_affine_qick_parameter(consumer, constant):
    linear, _ = time_program([0.0, 0.03125, 0.0625], linear=True, consumer=consumer)
    expected = np.asarray(linear.get_time_param("time", "t", as_array=True)).flatten()
    if constant:
        expected = np.repeat(expected[0], len(expected))
    # Use a bare delay oracle: pulse timeline placement already reports the same
    # physical parameter, while timeline_inc reports its rounded endpoint.
    table, _ = time_program(expected.tolist(), table=True, consumer=consumer)
    actual = table.get_time_param("time", "t")
    assert not isinstance(actual, TableParam)
    if not constant:
        assert isinstance(actual, QickParam)
    points = table.get_time_param("time", "t", as_array=True)
    assert np.array_equal(
        np.asarray(points).flatten(), expected if not constant else expected[:1]
    )


@pytest.mark.parametrize("consumer", ["delay", "timeline_inc", "native_t", "pulse_t"])
def test_nonaffine_table_time_getter_raises(consumer):
    prog, _ = time_program([0.0213, 0.0107, 0.0354], table=True, consumer=consumer)
    with pytest.raises(RuntimeError, match="non-affine"):
        prog.get_time_param("time", "t")


def test_register_time_table_rejects_negative_noninitial_point():
    with pytest.raises(ValueError, match="nonnegative"):
        time_program([0.02, -0.01, 0.03], table=True, consumer="register")
