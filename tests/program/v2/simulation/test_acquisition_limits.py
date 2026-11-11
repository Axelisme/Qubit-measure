"""Hardware upload limits at the public offline acquisition boundary."""

from collections.abc import Iterator
from typing import Literal

import numpy as np
import pytest
from zcu_tools.experiment.simulation import SimulationEnvironment, white_noise
from zcu_tools.program.v2 import TimelineProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._profiles import load_soccfg
from tests.program._timeline import compile_modules


@pytest.fixture(params=[4, 7])
def environment(request: pytest.FixtureRequest) -> Iterator[SimulationEnvironment]:
    soccfg = load_soccfg()
    soccfg["gens"][0]["maxlen"] = request.param * soccfg["gens"][0]["samps_per_clk"]
    soccfg["readouts"][0]["buf_maxlen"] = request.param * 8
    env = white_noise(soccfg, std_adc=0, seed=12, routes=((0, 0),))
    try:
        yield env
    finally:
        env.devices.close_all_devices()


def program(
    env: SimulationEnvironment, *, samples: int = 8, reps: int = 1
) -> TimelineProgramV2:
    return compile_modules(
        env.soccfg,
        modules=[
            PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=100,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.1),
                ro_length=env.soccfg.cycles2us(samples, ro_ch=0),
            ).build("readout")
        ],
        reps=reps,
    )


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_acquire_rejects_envelope_end_past_generator_capacity(
    environment: SimulationEnvironment, mode: Literal["integrated", "decimated"]
):
    prog = program(environment)
    capacity = environment.soccfg["gens"][0]["maxlen"]
    block = environment.soccfg["gens"][0]["samps_per_clk"]
    # QICK assigns the second envelope a nonzero sample address. Each envelope
    # fits by itself; their combined upload extends one fabric cycle past RAM.
    prog.add_envelope(0, "first", idata=np.ones(capacity, dtype=np.int16))
    prog.add_envelope(0, "second", idata=np.ones(block, dtype=np.int16))
    acquire = prog.acquire if mode == "integrated" else prog.acquire_decimated
    with pytest.raises(RuntimeError, match="buffer length"):
        acquire(environment.soc, progress=False)


def test_decimated_rejects_window_equal_to_raw_buffer_capacity(
    environment: SimulationEnvironment,
):
    capacity = environment.soccfg["readouts"][0]["buf_maxlen"]
    prog = program(environment, samples=capacity)
    with pytest.raises(RuntimeError, match="longer or equal to decimated buffer size"):
        prog.acquire_decimated(environment.soc, progress=False)


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_acquire_allows_envelope_end_exactly_at_generator_capacity(
    environment: SimulationEnvironment, mode: Literal["integrated", "decimated"]
):
    prog = program(environment)
    capacity = environment.soccfg["gens"][0]["maxlen"]
    block = environment.soccfg["gens"][0]["samps_per_clk"]
    prog.add_envelope(0, "first", idata=np.ones(capacity - block, dtype=np.int16))
    prog.add_envelope(0, "second", idata=np.ones(block, dtype=np.int16))
    acquire = prog.acquire if mode == "integrated" else prog.acquire_decimated
    result = acquire(environment.soc, progress=False)
    np.testing.assert_array_equal(result, 0)


def test_integrated_skips_envelope_upload_checks_when_upload_is_disabled(
    environment: SimulationEnvironment,
):
    prog = program(environment)
    capacity = environment.soccfg["gens"][0]["maxlen"]
    block = environment.soccfg["gens"][0]["samps_per_clk"]
    prog.add_envelope(0, "unused", idata=np.ones(capacity + block, dtype=np.int16))
    result = prog.acquire(environment.soc, load_envelopes=False, progress=False)
    np.testing.assert_array_equal(result, 0)


def test_decimated_allows_window_one_sample_below_raw_buffer_capacity(
    environment: SimulationEnvironment,
):
    capacity = environment.soccfg["readouts"][0]["buf_maxlen"]
    prog = program(environment, samples=capacity - 1)
    result = prog.acquire_decimated(environment.soc, progress=False)
    assert np.asarray(result).shape == (1, capacity - 1, 2)
    np.testing.assert_array_equal(result, 0)


@pytest.mark.parametrize("samples_past_capacity", [0, 1])
def test_integrated_readout_is_not_limited_by_raw_buffer_capacity(
    environment: SimulationEnvironment, samples_past_capacity: int
):
    capacity = environment.soccfg["readouts"][0]["buf_maxlen"]
    prog = program(environment, samples=capacity + samples_past_capacity)
    result = prog.acquire(environment.soc, progress=False)
    assert np.asarray(result).shape == (1, 1, 2)
    np.testing.assert_array_equal(result, 0)


def test_decimated_retains_qick_total_capture_capacity_check(
    environment: SimulationEnvironment,
):
    capacity = environment.soccfg["readouts"][0]["buf_maxlen"]
    # Each window is legal. Repetitions make their combined capture too large.
    prog = program(environment, samples=capacity // 2 + 1, reps=2)
    with pytest.raises(RuntimeError, match="exceeds buffer size"):
        prog.acquire_decimated(environment.soc, progress=False)
