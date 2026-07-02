from collections.abc import Callable

import pytest
from zcu_tools.program.v2.modules.base import AbsModuleCfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    NoneResetCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import (
    AbsWaveformCfg,
    ArbWaveformCfg,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)

CfgFactory = Callable[[], AbsModuleCfg | AbsWaveformCfg]


def _pulse_cfg() -> PulseCfg:
    return PulseCfg(
        waveform=ConstWaveformCfg(length=0.2),
        ch=0,
        nqz=1,
        freq=5000.0,
        gain=0.3,
    )


def _direct_readout_cfg() -> DirectReadoutCfg:
    return DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=6000.0)


@pytest.mark.parametrize(
    "make_cfg",
    [
        lambda: ConstWaveformCfg(length=1.0),
        lambda: CosineWaveformCfg(length=1.0),
        lambda: GaussWaveformCfg(length=1.0, sigma=0.2),
        lambda: DragWaveformCfg(length=1.0, sigma=0.2, delta=0.1, alpha=0.0),
        lambda: ArbWaveformCfg(data="demo"),
        lambda: FlatTopWaveformCfg(
            length=1.0,
            raise_waveform=GaussWaveformCfg(length=0.2, sigma=0.05),
        ),
        _pulse_cfg,
        _direct_readout_cfg,
        lambda: PulseReadoutCfg(
            pulse_cfg=_pulse_cfg(),
            ro_cfg=_direct_readout_cfg(),
        ),
        lambda: NoneResetCfg(),
        lambda: PulseResetCfg(pulse_cfg=_pulse_cfg()),
        lambda: TwoPulseResetCfg(
            pulse1_cfg=_pulse_cfg(),
            pulse2_cfg=_pulse_cfg(),
        ),
        lambda: BathResetCfg(
            cavity_tone_cfg=_pulse_cfg(),
            qubit_tone_cfg=_pulse_cfg(),
            pi2_cfg=_pulse_cfg(),
        ),
    ],
)
def test_concrete_cfg_set_param_rejects_unknown_name(make_cfg: CfgFactory):
    cfg = make_cfg()

    with pytest.raises(ValueError):
        cfg.set_param("__unknown_param__", 1.0)
