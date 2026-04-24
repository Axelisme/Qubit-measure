import pytest
from qick.asm_v2 import QickParam
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.registry import PulseRegistry
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def _make_cfg(freq=5000.0, phase=0.0, gain=0.5, ch=0, length=0.1):
    return PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=ch,
        nqz=1,
        freq=freq,
        phase=phase,
        gain=gain,
    )


def test_register_new_pulse_returns_true():
    reg = PulseRegistry()
    assert reg.register("p0", _make_cfg()) is True


def test_register_duplicate_returns_false():
    reg = PulseRegistry()
    cfg = _make_cfg()
    reg.register("p0", cfg)
    assert reg.register("p0_dup", _make_cfg()) is False


def test_register_different_freq_is_new():
    reg = PulseRegistry()
    reg.register("p0", _make_cfg(freq=5000.0))
    assert reg.register("p1", _make_cfg(freq=5001.0)) is True


def test_calc_name_stable_with_qickparam():
    reg = PulseRegistry()
    cfg1 = _make_cfg(freq=QickParam(start=5000.0, spans={"s": 1.0}))
    cfg2 = _make_cfg(freq=QickParam(start=5000.0, spans={"s": 1.0}))
    assert reg.calc_name(cfg1) == reg.calc_name(cfg2)


def test_check_valid_mixer_freq_mismatch_present_vs_absent():
    reg = PulseRegistry()
    cfg_with_mixer = _make_cfg()
    cfg_with_mixer.mixer_freq = 100.0
    reg.register("p_with", cfg_with_mixer)

    cfg_no_mixer = _make_cfg()
    with pytest.raises(ValueError):
        reg.check_valid_mixer_freq("p_no", cfg_no_mixer)
