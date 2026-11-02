"""Length lookup sweeps preserve scalar wave words and scheduling per point."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import SweepCfg, TimelineBuilder, ir
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.reset import PulseReset, PulseResetCfg, Reset
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)

from tests.program._profiles import load_soccfg
from tests.program.v2.simulation._compiled_support import request

RAMP_LENGTH_US = 0.04


def length_program(
    value, *, mode="scalar", flat=False, consumer="pulse", cfg_length=False
):
    env = white_noise(load_soccfg(2, 1), std_adc=0, routes=((0, 0),), seed=None)
    builder = TimelineBuilder(env.soccfg, reps=2, initial_delay=0, relax_delay=0.1)
    if mode == "table":
        value = builder.add_table_sweep("point", value)
    elif mode == "linear":
        value = builder.add_sweep(
            "point",
            SweepCfg(
                start=value[0],
                stop=value[-1],
                expts=len(value),
                step=(value[-1] - value[0]) / (len(value) - 1),
            ),
        )
    waveform = (
        FlatTopWaveformCfg(
            length=value if cfg_length else 0.2,
            raise_waveform=GaussWaveformCfg(length=RAMP_LENGTH_US, sigma=0.01),
        )
        if flat
        else ConstWaveformCfg(length=value if cfg_length else 0.1)
    )
    pulse_cfg = PulseCfg(ch=1, nqz=1, freq=80, gain=0.2, waveform=waveform)
    override = None if cfg_length else value
    receiver = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=waveform if consumer == "readout" else ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )
    if consumer == "readout":
        module = PulseReadout("readout", receiver, length=override)
        pulse = module.pulse
    elif consumer == "reset":
        module = Reset("reset", PulseResetCfg(pulse_cfg=pulse_cfg), length=override)
        assert isinstance(module, PulseReset)
        pulse = module.reset_pulse
    else:
        module = pulse = Pulse("drive", pulse_cfg, length=override)
    builder.emit(module)
    if consumer != "readout":
        builder.emit(receiver.build("readout"))
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


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("consumer", ["pulse", "readout", "reset"])
@pytest.mark.parametrize("cfg_length", [False, True])
def test_length_matches_scalar_words_timing_and_linear_actual_points(
    flat, consumer, cfg_length
):
    values = [0.213, 0.107, 0.354]
    options = dict(flat=flat, consumer=consumer, cfg_length=cfg_length)
    prog, env, _ = length_program(values, mode="table", **options)
    observed = request(prog, env)
    expected_shots = []
    for index, value in enumerate(values):
        scalar, scalar_env, _ = length_program(value, **options)
        assert replace(observed.materialize_point(index), coordinates=()) == request(
            scalar, scalar_env
        ).materialize_point(0)
        expected_shots.append(scheduled_shots(scalar)[0])
    assert scheduled_shots(prog) == expected_shots * 2

    linear, linear_env, name = length_program(values, mode="linear", **options)
    actual = np.asarray(linear.get_pulse_param(name, "length", as_array=True)).flatten()
    # QICK's flat-top length parameter is the flat segment, whereas our cfg
    # includes the authored ramp. Undo only that public input transformation.
    if flat:
        actual = actual + RAMP_LENGTH_US
    scalar_shots = []
    for index, value in enumerate(actual):
        scalar, scalar_env, _ = length_program(float(value), **options)
        scalar_shots.append(scheduled_shots(scalar)[0])
        assert replace(
            request(linear, linear_env).materialize_point(index), coordinates=()
        ) == request(scalar, scalar_env).materialize_point(0)
    # Only re-encodable actual points may serve as the linear parity oracle.
    assert scheduled_shots(linear) == scalar_shots * 2
    table, table_env, _ = length_program(actual.tolist(), mode="table", **options)
    assert scheduled_shots(table) == scheduled_shots(linear)
    for index in range(len(actual)):
        assert request(table, table_env).materialize_point(index) == request(
            linear, linear_env
        ).materialize_point(index)
