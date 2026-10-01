from __future__ import annotations

from zcu_tools.experiment.v2.lookback import LookbackCfg, LookbackModuleCfg
from zcu_tools.program.v2 import PulseCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def make_lookback_cfg(*, reps: int = 1, trig_offset: float = 0.4) -> LookbackCfg:
    readout = PulseReadoutCfg(
        pulse_cfg=PulseCfg(
            ch=0,
            nqz=1,
            freq=6000.0,
            gain=1.0,
            waveform=ConstWaveformCfg(length=2.0),
        ),
        ro_cfg=DirectReadoutCfg(
            ro_ch=0, gen_ch=0, ro_length=2.0, ro_freq=6000.0, trig_offset=trig_offset
        ),
    )
    return LookbackCfg(reps=reps, rounds=1, modules=LookbackModuleCfg(readout=readout))
