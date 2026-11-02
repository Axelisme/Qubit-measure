"""Wave-field lookup sweeps through public build, simulation and compiled words."""

from dataclasses import replace

import numpy as np
import pytest
from qick.asm_v2 import QickSweep1D
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import SweepCfg, TimelineBuilder, ir
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)

from tests.program._profiles import load_soccfg
from tests.program.v2.simulation._compiled_support import request


def field_program(
    field,
    value,
    *,
    mode="scalar",
    flat=False,
    paired=False,
    mixer=None,
    outer=False,
    branch=False,
    mixed=False,
):
    soccfg = load_soccfg(2, 1)
    soccfg["gens"][0]["has_mixer"] = mixer is not None
    env = white_noise(soccfg, std_adc=0, routes=((0, 0),), seed=None)
    builder = TimelineBuilder(env.soccfg, reps=2, initial_delay=0, relax_delay=0.1)
    if outer:
        builder.add_loop("outer", 2)
    if mode == "table":
        parameter = builder.add_table_sweep("point", value)
    elif mode == "linear":
        parameter = builder.add_sweep(
            "point",
            SweepCfg(
                start=value[0],
                stop=value[-1],
                expts=len(value),
                step=(value[-1] - value[0]) / (len(value) - 1),
            ),
        )
    else:
        parameter = value
    knobs = {"freq": 100.0 if paired else 80.0, "phase": 0.0, "gain": 0.2}
    knobs[field] = parameter
    if mixed:
        knobs["gain"] = QickSweep1D("point", 0.1, 0.3)
    waveform = (
        FlatTopWaveformCfg(
            length=0.2,
            raise_waveform=GaussWaveformCfg(length=0.04, sigma=0.01),
        )
        if flat
        else ConstWaveformCfg(length=0.1)
    )
    receiver = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=waveform if paired else ConstWaveformCfg(length=0.1),
        ro_length=0.1,
        mixer_freq=mixer,
    )
    if paired:
        module = PulseReadout("readout", receiver, **knobs)
        pulse = module.pulse
    else:
        pulse = Pulse(
            "drive",
            PulseCfg(
                ch=1,
                nqz=1,
                freq=80,
                gain=0.2,
                waveform=waveform,
            ),
            **knobs,
        )
        module = pulse
    if branch:
        with builder.branch(by="point", align_timeline=True) as cases, cases.case(2):
            builder.emit(module)
    else:
        builder.emit(module)
    if not paired:
        builder.emit(receiver.build("readout"))
    builder.delay_by_timeline()
    prog = builder.build()
    return prog, env, pulse.pulse_name(prog)


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


def assert_scalar_points(prog, env, values, field, **options):
    observed = request(prog, env)
    scalar_words = []
    for index, value in enumerate(values):
        scalar, scalar_env, _ = field_program(field, value, **options)
        # Coordinates identify the scan, not the scalar hardware output.
        assert replace(observed.materialize_point(index), coordinates=()) == request(
            scalar, scalar_env
        ).materialize_point(0)
        scalar_words.append(executed_words(scalar)[0])
    assert executed_words(prog) == scalar_words * 2


def test_const_frequency_table_matches_scalar_program_at_every_point_and_rep():
    values = [83.125, 76.75, 91.5]
    prog, env, _ = field_program("freq", values, mode="table")
    assert_scalar_points(prog, env, values, "freq")


def test_readout_phase_override_table_matches_scalar_program():
    values = [10.5, -30.25, 80.75]
    prog, env, _ = field_program("phase", values, mode="table", paired=True)
    assert_scalar_points(prog, env, values, "phase", paired=True)


@pytest.mark.parametrize(
    "field,values",
    [
        ("freq", [83.125, 76.75, 91.5]),
        ("phase", [10.5, -30.25, 80.75]),
        ("gain", [0.2, -0.1, 0.3]),
    ],
)
@pytest.mark.parametrize(
    "flat,paired,mixer",
    [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (False, True, 50.25),
        (True, True, None),
        (True, True, 50.25),
    ],
)
def test_wave_tables_match_scalar_words_and_actual_linear_points(
    field, values, flat, paired, mixer
):
    options = dict(flat=flat, paired=paired, mixer=mixer)
    prog, env, _ = field_program(field, values, mode="table", **options)
    assert_scalar_points(prog, env, values, field, **options)
    linear, linear_env, name = field_program(field, values, mode="linear", **options)
    actual = (
        np.asarray(linear.get_pulse_param(name, field, as_array=True))
        .flatten()
        .tolist()
    )
    table, table_env, _ = field_program(field, actual, mode="table", **options)
    linear_request = request(linear, linear_env)
    table_request = request(table, table_env)
    for point in range(len(actual)):
        assert table_request.materialize_point(
            point
        ) == linear_request.materialize_point(point)
    assert executed_words(table) == executed_words(linear)
