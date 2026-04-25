import pytest
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def _make_cfg(length=0.2, pre=0.0, post=0.0):
    return PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=0,
        nqz=1,
        freq=5000.0,
        phase=0.0,
        gain=0.3,
        pre_delay=pre,
        post_delay=post,
    )


def test_pulsecfg_validates_nqz():
    with pytest.raises(Exception):
        PulseCfg(
            waveform=ConstWaveformCfg(length=0.1),
            ch=0,
            nqz=3,  # type: ignore , only 1 or 2 allowed
            freq=5000.0,
            gain=0.3,
        )


def test_pulsecfg_set_param_gain_freq_phase():
    cfg = _make_cfg()
    cfg.set_param("gain", 0.9)
    cfg.set_param("freq", 5100.0)
    cfg.set_param("phase", 45.0)
    assert cfg.gain == 0.9
    assert cfg.freq == 5100.0
    assert cfg.phase == 45.0


def test_pulsecfg_set_param_length_delegates_to_waveform():
    cfg = _make_cfg(length=0.2)
    cfg.set_param("length", 0.5)
    assert cfg.waveform.length == 0.5


def test_pulsecfg_set_param_unknown_rejected():
    cfg = _make_cfg()
    with pytest.raises(ValueError):
        cfg.set_param("ch", 1)


def test_pulse_none_cfg_is_no_op(mock_prog):
    p = Pulse("p", None)
    assert p.run(mock_prog, t=0.5) == 0.5
    assert p.total_length(mock_prog) == 0.0
    assert p.allow_rerun() is True
