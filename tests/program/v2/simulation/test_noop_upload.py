"""Simulation validates compiled uploads without serializing unused transport data."""

import numpy as np
import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


class ObservedData(np.ndarray):
    """Observe the external NumPy serialization boundary, not frontend internals."""

    def tolist(self):
        raise AssertionError("simulation must not serialize a no-op DAC upload")


def test_integrated_upload_skips_lists_and_still_rejects_invalid_dtype():
    env = white_noise(std_adc=0, seed=1)
    try:
        program = ModularProgramV2(
            env.soccfg,
            modules=[
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=100,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=0.1),
                    ro_length=0.1,
                ).build("readout")
            ],
            reps=2,
        )
        # Extra compiled DAC data is irrelevant to this const readout, but QICK
        # still validates it when uploads are requested.
        contents = program.envelopes[0]["envs"]
        assert isinstance(contents, dict)
        contents["extra"] = {
            "data": np.zeros((16, 2), dtype=np.int16).view(ObservedData),
            "addr": 0,
        }
        result = program.acquire(env.soc, progress=False)
        assert result is not None
        assert np.all(result[0] == 0)
        contents["extra"] = {"data": np.zeros((16, 2), dtype=np.float64), "addr": 0}
        with pytest.raises(AssertionError):
            program.acquire(env.soc, progress=False)
    finally:
        env.devices.close_all_devices()


def test_integrated_weight_upload_validates_without_serializing():
    env = white_noise(std_adc=0, seed=1)
    try:
        program = ModularProgramV2(
            env.soccfg,
            modules=[
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=100,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=0.1),
                    ro_length=0.1,
                ).build("readout")
            ],
            reps=2,
        )
        program.soccfg["readouts"][0]["has_weights"] = True
        program.ro_chs[0]["weights"] = np.zeros((8, 2), dtype=np.int16).view(
            ObservedData
        )
        result = program.acquire(env.soc, progress=False)
        assert result is not None
        assert np.all(result[0] == 0)
        program.ro_chs[0]["weights"] = np.zeros((8, 2), dtype=np.float64)
        with pytest.raises(AssertionError):
            program.acquire(env.soc, progress=False)
        # QICK's opt-out keeps the original contract: no upload validation.
        program.acquire(env.soc, progress=False, load_envelopes=False)
    finally:
        env.devices.close_all_devices()
