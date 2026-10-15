from collections.abc import Callable

import pytest
from zcu_tools.cfg_model import (
    AbsModuleCfg,
    AbsWaveformCfg,
    ArbWaveformCfg,
    BathResetCfg,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
    NoneResetCfg,
    PulseCfg,
    PulseReadoutCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
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
        lambda: PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            freq=100.0,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.5),
            ro_ch=0,
            ro_length=1.0,
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

    if isinstance(cfg, NoneResetCfg):
        error = "NoneReset does not support set_param"
    elif isinstance(cfg, ArbWaveformCfg):
        error = "Arb waveform length is asset-derived"
    else:
        error = "Unknown parameter: __unknown_param__"

    with pytest.raises(ValueError, match=error):
        cfg.set_param("__unknown_param__", 1.0)


@pytest.mark.parametrize("name", ["freq", "gain", "phase", "length", "ro_length", "tones.0.freq"])
def test_mux_set_param_requires_cfg_and_program_rebuild(name: str):
    from zcu_tools.cfg_model import ModuleCfgFactory

    from ._support import mux_raw

    cfg = ModuleCfgFactory.from_raw(mux_raw())
    before = cfg.to_dict()
    with pytest.raises(ValueError, match="rebuild.*cfg.*program"):
        cfg.set_param(name, 2.0)
    assert cfg.to_dict() == before
