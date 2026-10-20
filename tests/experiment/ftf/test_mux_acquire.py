"""Persisted mux settings drive the real program-to-FTF acquisition path."""

import numpy as np
import pytest
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.experiment.ftf import ftf
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import mux_document

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("channels", [(4, 1), (1,)])
@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_yaml_mux_program_produces_each_tones_raw_shots(channels, mode):
    try:
        settings = parse_ftf(mux_document(channels))
    except ValueError as error:
        pytest.fail(f"mux settings must enable end-to-end FTF acquisition: {error}")
    env = ftf(settings)
    try:
        module = MuxReadoutCfg(
            gen_ch=0,
            nqz=1,
            mixer_freq=6000.0,
            length=0.020,
            ro_length=0.026,
            tones=[
                MuxReadoutTone(ro_ch=ch, freq=6000.0 + 80 * i, gain=0.1, phase=45 * i)
                for i, ch in enumerate(channels)
            ],
        )
        prog = ModularProgramV2(
            env.soccfg, ProgramV2Cfg(reps=7), modules=[module.build("joint")]
        )
        if mode == "integrated":
            prog.acquire(env.soc, progress=False)
        else:
            prog.acquire_decimated(env.soc, progress=False)
        request = env.soc.binding.last_request
        assert request is not None
        assert tuple(w.channel for w in request.readouts) == channels
        raw = prog.get_raw()
        assert raw is not None
        assert len(raw) == len(channels)
        for shots in raw:
            assert shots.dtype == np.int64
            assert shots.size == 14
            assert np.any(shots != 0)
        again = prog.get_raw()
        assert again is not None
        for first, second in zip(raw, again, strict=True):
            np.testing.assert_array_equal(first, second)
    finally:
        env.devices.close_all_devices()
