"""Explicit environments exercise the real QICK acquisition/packing pipeline."""

import importlib
import importlib.util
from threading import Event

import numpy as np
import pytest
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def environment(**kwargs):
    name = "zcu_tools.experiment.simulation"
    assert importlib.util.find_spec(name) is not None, (
        "explicit environment API required"
    )
    return importlib.import_module(name).white_noise(**kwargs)


def program(env, *, reps=2, rounds=1, channels=(0,), samples=16):
    return ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(reps=reps, rounds=rounds),
        modules=[
            PulseReadoutCfg(
                gen_ch=ch,
                ro_ch=ch,
                nqz=1,
                freq=100.0,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.1),
                ro_length=env.soccfg.cycles2us(samples, ro_ch=ch),
            ).build(f"readout_{ch}")
            for ch in channels
        ],
    )


def raw_of(prog):
    raw = prog.get_raw()
    if raw is None:
        raise AssertionError("completed acquisition must retain raw buffers")
    return raw


def test_explicit_zero_noise_preserves_shapes_and_raw():
    env = environment(std_adc=0, seed=123)
    prog = program(env, rounds=2)
    result = prog.acquire(env.soc, progress=False)
    assert result[0].shape == (1, 2)
    np.testing.assert_array_equal(result[0], 0)
    raw = prog.get_raw()
    assert raw is not None
    assert raw[0].shape == (2, 1, 2)
    assert raw[0].dtype == np.int64
    dec = prog.acquire_decimated(env.soc, progress=False)
    assert dec[0].shape == (2, 16, 2)
    np.testing.assert_array_equal(dec[0], 0)


def test_gaussian_sum_scaling_and_reproducible_independent_acquires():
    # 12000 independent quadratures: mean/std bounds exceed six sampling SEs.
    env = environment(std_adc=20, seed=41)
    prog = program(env, reps=6000, samples=9)
    prog.acquire(env.soc, progress=False)
    first = raw_of(prog)[0].copy()
    assert abs(first.mean()) < 2
    assert 58 < first.std() < 62  # sqrt(9) * 20 ADC sum units
    prog.acquire(env.soc, progress=False)
    assert not np.array_equal(first, raw_of(prog)[0])
    repeat = environment(std_adc=20, seed=41)
    replay = program(repeat, reps=6000, samples=9)
    replay.acquire(repeat.soc, progress=False)
    np.testing.assert_array_equal(first, raw_of(replay)[0])


def test_decimated_companion_raw_uses_same_trace_and_adc_order():
    env = environment(std_adc=3, seed=91, n_gens=2, n_readouts=2)
    prog = program(env, channels=(1, 0), reps=3, samples=7)
    traces = prog.acquire_decimated(env.soc, progress=False)
    raw = raw_of(prog)
    for trace, accumulated in zip(traces, raw, strict=True):
        np.testing.assert_array_equal(
            accumulated[:, 0, :], np.rint(trace.sum(axis=-2)).astype(np.int64)
        )
    assert not np.array_equal(traces[0], traces[1])
    for ch, trace in zip((1, 0), traces, strict=True):
        np.testing.assert_array_equal(
            env.soc.get_decimated(ch, 0, 21), trace.reshape(-1, 2)
        )
        np.testing.assert_array_equal(
            env.soc.get_decimated(ch, 0, 21), trace.reshape(-1, 2)
        )


@pytest.mark.parametrize("mode", ["acquire", "acquire_decimated"])
def test_step_rounds_rejected_before_consuming_sequence(mode):
    env = environment(seed=11)
    prog = program(env)
    with pytest.raises(ValueError, match="step_rounds"):
        getattr(prog, mode)(env.soc, progress=False, step_rounds=True)
    getattr(prog, mode)(env.soc, progress=False)
    assert env.soc.binding.last_request.sequence == 0


def test_reentry_does_not_corrupt_original_rounds_and_stop_consumes_sequence():
    env = environment(seed=21)
    prog = program(env, rounds=3)
    observed = []

    def hook(count, data, stop):
        observed.append(count)
        with pytest.raises(RuntimeError, match="active|reentr"):
            prog.acquire_decimated(env.soc, progress=False)
        if count == 2:
            stop.set()

    prog.acquire(env.soc, progress=False, round_hook=hook)
    assert observed == [1, 2]
    rounds = prog.get_rounds()
    assert rounds is not None and len(rounds) == 2
    prog.acquire(env.soc, progress=False)
    assert env.soc.binding.last_request.sequence == 1
    stop = Event()
    stop.set()
    with pytest.raises(StoppedPartialAcquireError):
        prog.acquire(env.soc, progress=False, cancel_flag=stop)
    prog.acquire(env.soc, progress=False)
    assert env.soc.binding.last_request.sequence == 3


@pytest.mark.parametrize("std_adc", [-1, float("inf"), float("nan")])
def test_invalid_std_rejected(std_adc):
    with pytest.raises(ValueError, match="std_adc"):
        environment(std_adc=std_adc)


def test_unbound_channel_rejected():
    env = environment(n_gens=2, n_readouts=2, routes=((0, 0),))
    prog = program(env, channels=(1,))
    with pytest.raises(ValueError, match="bound|route"):
        prog.acquire(env.soc, progress=False)
