"""Time lookup sweeps preserve scalar scheduling at each compiled scan point."""

from dataclasses import replace

import pytest

from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import TimelineBuilder, ir
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._profiles import load_soccfg
from tests.program.v2.simulation._compiled_support import request


def time_program(value, *, table=False, consumer="delay"):
    env = white_noise(load_soccfg(1, 1), std_adc=0, routes=((0, 0),), seed=None)
    builder = TimelineBuilder(env.soccfg, reps=2, initial_delay=0, relax_delay=0.1)
    if table:
        value = builder.add_table_sweep("point", value)
    if consumer == "timeline_inc":
        builder.timeline_inc(value)
    elif consumer == "delay":
        builder.delay(value)
    builder.emit(
        PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=100,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=0.1,
            pre_delay=value if consumer == "pre_delay" else 0.0,
        ).build("readout")
    )
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


@pytest.mark.parametrize("consumer", ["delay", "timeline_inc", "pre_delay"])
def test_time_table_matches_scalar_at_every_point_and_rep(consumer):
    values = [0.213, 0.107, 0.354]
    prog, env = time_program(values, table=True, consumer=consumer)
    observed = request(prog, env)
    expected_shots = []
    for point, value in enumerate(values):
        scalar, scalar_env = time_program(value, consumer=consumer)
        assert replace(observed.materialize_point(point), coordinates=()) == request(
            scalar, scalar_env
        ).materialize_point(0)
        expected_shots.append(scheduled_shots(scalar)[0])
    assert scheduled_shots(prog) == expected_shots * 2
