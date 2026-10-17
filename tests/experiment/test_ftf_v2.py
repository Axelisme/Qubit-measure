"""V2 channel derivation, live DC and charge-drive acquisition seam."""

import numpy as np
import pytest
from zcu_tools.device.fake import FakeDevice
from zcu_tools.experiment.ftf import ftf
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import ftf_v2_document


def test_v2_derives_sparse_channels_and_acquires_with_live_dc() -> None:
    document = ftf_v2_document()
    # Use a readout hole as well as the fixture's generator hole.
    readout = document["readout"]
    assert isinstance(readout, dict)
    readout["adc"]["ro_ch"] = 2
    try:
        settings = parse_ftf(document)
    except ValueError as error:
        pytest.fail(f"v2 sparse channel document must validate: {error}")
    env = ftf(settings)
    try:
        assert len(env.soccfg["gens"]) == 3
        assert len(env.soccfg["readouts"]) == 3
        driver = env.devices.get_device("bias")
        if not isinstance(driver, FakeDevice):
            raise AssertionError("FTF DC source must be a fixed-unit memory device")
        assert driver.get_info().value_unit == "V"
        assert driver.get_value() == 0.2
        pulse = PulseCfg(
            ch=2,
            nqz=1,
            freq=250.0,
            phase=90.0,
            gain=-0.2,
            waveform=ConstWaveformCfg(length=0.008),
        )
        ro = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=2,
            nqz=1,
            freq=6000.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.016),
            ro_length=0.016,
        )
        program = ModularProgramV2(
            env.soccfg,
            ProgramV2Cfg(reps=2, rounds=2),
            modules=[pulse.build("charge"), ro.build("readout")],
        )

        def change_bias(*_args, **_kwargs) -> None:
            driver.set_value(0.4)

        result = program.acquire_decimated(
            env.soc, progress=False, round_hook=change_bias
        )
        assert np.isfinite(result[0]).all()
        assert np.any(np.asarray(result[0]) != 0)
        assert [r.snapshot.values for r in env.soc.binding.rounds] == [
            (("bias", 0.2),),
            (("bias", 0.4),),
        ]
        unbound = ModularProgramV2(
            env.soccfg,
            ProgramV2Cfg(reps=1),
            modules=[pulse.with_updates(ch=1).build("unbound"), ro.build("readout")],
        )
        with pytest.raises(ValueError, match="unbound|route"):
            unbound.acquire(env.soc, progress=False)
    finally:
        env.devices.close_all_devices()
