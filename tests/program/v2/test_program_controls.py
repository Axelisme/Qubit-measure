"""Explicit program repetition and timing controls at the compiled public seam."""

from tests.program._profiles import load_soccfg
import numpy as np
import pytest
from zcu_tools.experiment.simulation import white_noise

from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._timeline import compile_modules
from tests.program.v2.simulation.time_observations import time_us


def test_constructor_controls_define_repetition_loop_and_delays():
    prog = compile_modules(
        load_soccfg(), modules=[], reps=3, initial_delay=0.25, relax_delay=12.5
    )
    assert prog.loop_dict["reps"] == 3
    assert prog.initial_delay == 0.25
    assert prog.relax_delay == 12.5


@pytest.mark.parametrize("reps", [0, -1, True, 1.5])
def test_constructor_rejects_non_positive_integer_reps(reps):
    with pytest.raises(ValueError, match="reps must be a positive integer"):
        compile_modules(load_soccfg(), modules=[], reps=reps)


@pytest.mark.parametrize("decimated", [False, True])
def test_acquire_uses_each_calls_rounds_and_preserves_iq_layout(decimated):
    env = white_noise(load_soccfg(), std_adc=0, seed=123, routes=((0, 0),))
    try:
        module = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=100.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=env.soccfg.cycles2us(16, ro_ch=0),
        ).build("readout")
        prog = compile_modules(env.soccfg, modules=[module], reps=3, relax_delay=12.5)
        acquire = prog.acquire_decimated if decimated else prog.acquire
        for rounds in (2, 3):
            completed = []
            result = acquire(
                env.soc,
                rounds=rounds,
                progress=False,
                round_hook=lambda count, raw, stop, completed=completed: (
                    completed.append(count)
                ),
            )
            assert completed == list(range(1, rounds + 1))
            assert result[0].shape == ((3, 16, 2) if decimated else (1, 2))
            np.testing.assert_array_equal(result[0], 0)
            request = env.soc.binding.last_request
            assert request is not None
            assert time_us(request.time_base, request.relax_delay_ticks) == 12.5
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("decimated", [False, True])
@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_acquire_rejects_non_positive_integer_rounds(decimated, rounds):
    prog = compile_modules(load_soccfg(), modules=[])
    acquire = prog.acquire_decimated if decimated else prog.acquire
    with pytest.raises(ValueError, match="rounds must be a positive integer"):
        acquire(object(), rounds=rounds)
