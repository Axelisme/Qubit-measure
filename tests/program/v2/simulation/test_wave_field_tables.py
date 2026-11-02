"""Wave-field lookup sweeps through public build, simulation and compiled words."""

from dataclasses import replace

from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import TimelineBuilder, ir
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._profiles import load_soccfg
from tests.program.v2.simulation._compiled_support import request


def const_program(value, *, swept=False):
    env = white_noise(load_soccfg(2, 1), std_adc=0, routes=((0, 0),), seed=None)
    builder = TimelineBuilder(env.soccfg, reps=2, initial_delay=0, relax_delay=0.1)
    frequency = builder.add_table_sweep("point", value) if swept else value
    builder.emit(
        Pulse(
            "drive",
            PulseCfg(
                ch=1, nqz=1, freq=80, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
            ),
            freq=frequency,
        )
    )
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
    return builder.build(), env


def executed_words(prog):
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
        tuple(e.wave_words for e in shot.events if e.wave_words is not None)
        for shot in analysis.shots
    ]


def test_const_frequency_table_matches_scalar_program_at_every_point_and_rep():
    values = [83.125, 76.75, 91.5]
    prog, env = const_program(values, swept=True)
    observed = request(prog, env)
    scalar_words = []
    for index, value in enumerate(values):
        scalar, scalar_env = const_program(value)
        # Coordinates identify the scan, not the scalar hardware output.
        assert replace(observed.materialize_point(index), coordinates=()) == request(
            scalar, scalar_env
        ).materialize_point(0)
        scalar_words.append(executed_words(scalar)[0])
    assert executed_words(prog) == scalar_words * 2
