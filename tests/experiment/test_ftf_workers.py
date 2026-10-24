"""Worker selection and numerical parity at public FTF composition/acquire seams."""

import numpy as np
import pytest
from qick.asm_v2 import QickSweep1D
from zcu_tools.experiment.ftf import ftf, runtime
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf
from zcu_tools.simulate.evolution import Engine

from tests.resources._ftf_data import ftf_document

pytestmark = pytest.mark.sim


def test_ftf_workers_reach_engine_and_preserve_small_acquire(monkeypatch):
    widths = []

    class ObservedEngine(Engine):
        def __init__(self, workers, cache_bytes, **kwargs):
            widths.append(workers)
            super().__init__(workers, cache_bytes, **kwargs)

    monkeypatch.setattr(runtime, "Engine", ObservedEngine)
    settings = parse_ftf(ftf_document())
    results = []
    for workers in (None, 2):
        try:
            env = ftf(settings) if workers is None else ftf(settings, workers=workers)
        except TypeError as error:
            pytest.fail(f"ftf must accept and forward keyword workers: {error}")
        try:
            pulse = PulseCfg(
                ch=2,
                nqz=1,
                freq=QickSweep1D("freq", 240.0, 260.0),
                gain=0.02,
                waveform=ConstWaveformCfg(length=0.008),
            )
            readout = PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=6000.0,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.016),
                ro_length=0.016,
            )
            program = ModularProgramV2(
                env.soccfg,
                modules=[pulse.build("drive"), readout.build("ro")],
                sweep=[("freq", 2)],
                reps=2,
            )
            acquired = program.acquire_decimated(env.soc, progress=False)
            assert acquired is not None
            results.append(acquired[0])
        finally:
            env.devices.close_all_devices()
    assert widths == [1, 2]
    np.testing.assert_allclose(results[1], results[0], atol=1e-12, rtol=0)
