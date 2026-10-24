"""Explicit program repetition and timing controls at the compiled public seam."""

import numpy as np
import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, make_mock_soccfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def test_constructor_controls_define_repetition_loop_and_delays():
    prog = ModularProgramV2(
        make_mock_soccfg(), modules=[], reps=3, initial_delay=0.25, relax_delay=12.5
    )
    assert prog.loop_dict["reps"] == 3
    assert prog.initial_delay == 0.25
    assert prog.relax_delay == 12.5


@pytest.mark.parametrize("reps", [0, -1, True, 1.5])
def test_constructor_rejects_non_positive_integer_reps(reps):
    with pytest.raises(ValueError, match="reps must be a positive integer"):
        ModularProgramV2(make_mock_soccfg(), modules=[], reps=reps)


@pytest.mark.parametrize("decimated", [False, True])
def test_acquire_uses_each_calls_rounds_and_preserves_iq_layout(decimated):
    env = white_noise(std_adc=0, seed=123)
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
        prog = ModularProgramV2(env.soccfg, modules=[module], reps=3, relax_delay=12.5)
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
            assert env.soc.binding.last_request is not None
            assert env.soc.binding.last_request.relax_delay_us == 12.5
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("decimated", [False, True])
@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_acquire_rejects_non_positive_integer_rounds(decimated, rounds):
    prog = ModularProgramV2(make_mock_soccfg(), modules=[])
    acquire = prog.acquire_decimated if decimated else prog.acquire
    with pytest.raises(ValueError, match="rounds must be a positive integer"):
        acquire(object(), rounds=rounds)
