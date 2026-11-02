"""Length lookup sweeps preserve scalar wave words and scheduling per point."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import SweepCfg, TimelineBuilder
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.reset import PulseReset, PulseResetCfg, Reset
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)

from tests.program._profiles import load_soccfg
from tests.program.v2.simulation._compiled_support import request, scheduled_shots

RAMP_LENGTH_US = 0.04


class AutoFollower(Module):
    """Play a scalar pulse after the preceding output on the same generator."""

    name = "follower"

    def __init__(self, ch):
        self.pulse = PulseCfg(
            ch=ch, nqz=1, freq=80, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
        ).build(self.name)
        self.ch = ch

    def declare(self, prog):
        self.pulse.declare(prog)

    def lower(self, prog, *, timeline, tag):
        prog.pulse(self.ch, self.pulse.pulse_name(prog), t="auto", timeline=timeline)


def length_program(
    value,
    *,
    mode="scalar",
    flat=False,
    consumer="pulse",
    cfg_length=False,
    outer=False,
    branch=False,
    point_count=1,
    auto=False,
    followers=1,
    pre_delay=0.0,
    post_delay=0.0,
):
    env = white_noise(load_soccfg(2, 1), std_adc=0, routes=((0, 0),), seed=None)
    builder = TimelineBuilder(env.soccfg, reps=2, initial_delay=0, relax_delay=0.1)
    if outer:
        builder.add_loop("outer", 2)
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
    if mode == "scalar" and branch:
        builder.add_loop("point", point_count)
    waveform = (
        FlatTopWaveformCfg(
            length=value if cfg_length else 0.2,
            raise_waveform=GaussWaveformCfg(length=RAMP_LENGTH_US, sigma=0.01),
        )
        if flat
        else ConstWaveformCfg(length=value if cfg_length else 0.1)
    )
    pulse_cfg = PulseCfg(
        ch=1,
        nqz=1,
        freq=80,
        gain=0.2,
        waveform=waveform,
        pre_delay=pre_delay,
        post_delay=post_delay,
    )
    override = None if cfg_length else value
    receiver = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=waveform if consumer == "readout" else ConstWaveformCfg(length=0.1),
        ro_length=0.1,
        pre_delay=pre_delay if consumer == "readout" else 0.0,
        post_delay=post_delay if consumer == "readout" else 0.0,
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
    if branch:
        with builder.branch(by="point", align_timeline=True) as cases, cases.case(2):
            builder.emit(module)
    else:
        builder.emit(module)
    if auto:
        for _ in range(followers):
            builder.emit(AutoFollower(0 if consumer == "readout" else 1))
    if consumer != "readout":
        builder.emit(receiver.build("readout"))
    builder.delay_by_timeline()
    prog = builder.build()
    return prog, env, pulse.pulse_name(prog)


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


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("branch", [False, True])
def test_length_wrap_and_unplayed_branch_keep_auto_follower_at_actual_end(flat, branch):
    values = [0.213, 0.107, 0.354]
    prog, env, _ = length_program(
        values, mode="table", flat=flat, outer=True, branch=branch, auto=True
    )
    observed = request(prog, env)
    expected_shots = []
    for index, value in enumerate(values):
        scalar, scalar_env, _ = length_program(
            value, flat=flat, branch=branch, point_count=len(values), auto=True
        )
        scalar_index = index if branch else 0
        expected = replace(
            request(scalar, scalar_env).materialize_point(scalar_index), coordinates=()
        )
        for outer in range(2):
            assert (
                replace(
                    observed.materialize_point(outer * len(values) + index),
                    coordinates=(),
                )
                == expected
            )
        expected_shots.append(scheduled_shots(scalar)[scalar_index])
    assert scheduled_shots(prog) == expected_shots * 4


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("consumer", ["pulse", "readout", "reset"])
def test_length_with_fractional_pre_and_post_matches_scalar_aggregate(flat, consumer):
    values = [0.213, 0.107, 0.354]
    prog, env, _ = length_program(
        values,
        mode="table",
        flat=flat,
        consumer=consumer,
        pre_delay=0.0137,
        post_delay=0.0189,
    )
    observed = request(prog, env)
    expected_shots = []
    for index, value in enumerate(values):
        scalar, scalar_env, _ = length_program(
            value, flat=flat, consumer=consumer, pre_delay=0.0137, post_delay=0.0189
        )
        assert replace(observed.materialize_point(index), coordinates=()) == request(
            scalar, scalar_env
        ).materialize_point(0)
        expected_shots.append(scheduled_shots(scalar)[0])
    assert scheduled_shots(prog) == expected_shots * 2


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("branch", [False, True])
@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("followers", [2, 3])
def test_consecutive_native_auto_followers_match_scalar_timestamps(
    flat, branch, outer, followers
):
    values = [0.2, 0.107, 0.213]
    prog, _, _ = length_program(
        values,
        mode="table",
        outer=outer,
        flat=flat,
        branch=branch,
        auto=True,
        followers=followers,
    )
    expected_shots = []
    for index, value in enumerate(values):
        scalar, _, _ = length_program(
            value,
            point_count=len(values),
            flat=flat,
            branch=branch,
            auto=True,
            followers=followers,
        )
        expected_shots.append(scheduled_shots(scalar)[index if branch else 0])
    assert scheduled_shots(prog) == expected_shots * (4 if outer else 2)


def test_branch_native_auto_followers_preserve_scalar_request():
    values = [0.2, 0.107, 0.213]
    prog, env, _ = length_program(
        values, mode="table", branch=True, auto=True, followers=2
    )
    observed = request(prog, env)
    for index, value in enumerate(values):
        scalar, scalar_env, _ = length_program(
            value, branch=True, point_count=len(values), auto=True, followers=2
        )
        assert replace(observed.materialize_point(index), coordinates=()) == replace(
            request(scalar, scalar_env).materialize_point(index), coordinates=()
        )


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("invalid_cycles", [2, 2**16])
def test_noninitial_length_point_outside_qick_generator_range_fails_build(
    flat, invalid_cycles
):
    soccfg = load_soccfg(2, 1)
    length = soccfg.cycles2us(invalid_cycles, gen_ch=1)
    if flat:
        length += RAMP_LENGTH_US
    with pytest.raises(RuntimeError, match="Pulse length"):
        length_program([0.2, length, 0.3], mode="table", flat=flat)
