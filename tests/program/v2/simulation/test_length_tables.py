"""Length lookup sweeps preserve scalar wave words and scheduling per point."""

from dataclasses import replace

from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import TimelineBuilder, ir
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._profiles import load_soccfg
from tests.program.v2.simulation._compiled_support import request


def length_program(value, *, table=False):
    env = white_noise(load_soccfg(2, 1), std_adc=0, routes=((0, 0),), seed=None)
    builder = TimelineBuilder(env.soccfg, reps=2, initial_delay=0, relax_delay=0.1)
    if table:
        value = builder.add_table_sweep("point", value)
    pulse = Pulse(
        "drive",
        PulseCfg(ch=1, nqz=1, freq=80, gain=0.2, waveform=ConstWaveformCfg(length=0.1)),
        length=value,
    )
    builder.emit(pulse)
    builder.emit(
        PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=100,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=0.1,
        ).build("readout")
    )
    builder.delay_by_timeline()
    prog = builder.build()
    return prog, env, pulse.pulse_name(prog)


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


def test_const_length_override_matches_scalar_words_and_timing():
    values = [0.213, 0.107, 0.354]
    prog, env, _ = length_program(values, table=True)
    observed = request(prog, env)
    expected_shots = []
    for index, value in enumerate(values):
        scalar, scalar_env, _ = length_program(value)
        assert replace(observed.materialize_point(index), coordinates=()) == request(
            scalar, scalar_env
        ).materialize_point(0)
        expected_shots.append(scheduled_shots(scalar)[0])
    assert scheduled_shots(prog) == expected_shots * 2
