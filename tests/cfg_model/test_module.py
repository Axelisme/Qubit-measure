"""Core module validation and runtime knob contracts."""

import pytest
from pydantic import TypeAdapter, ValidationError
from zcu_tools.cfg_model import (
    CfgInputMap,
    CfgInputValue,
    ConstWaveformCfg,
    DirectReadoutCfg,
    ModuleCfgFactory,
    PulseCfg,
    PulseReadoutCfg,
    ReadoutCfg,
)

_READOUT_ADAPTER: TypeAdapter[ReadoutCfg] = TypeAdapter(ReadoutCfg)


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
    with pytest.raises(ValidationError, match="nqz"):
        PulseCfg.model_validate(
            {
                "waveform": ConstWaveformCfg(length=0.1),
                "ch": 0,
                "nqz": 3,
                "freq": 5000.0,
                "gain": 0.3,
            }
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
    with pytest.raises(ValueError, match="Unknown parameter: ch"):
        cfg.set_param("ch", 1)


def _direct_dict(**kw: CfgInputValue) -> CfgInputMap:
    base: CfgInputMap = {
        "type": "readout/direct",
        "ro_ch": 0,
        "ro_length": 1.0,
        "ro_freq": 5000.0,
    }
    base.update(kw)
    return base


def _pulse_dict(**kw: CfgInputValue) -> CfgInputMap:
    d: CfgInputMap = {
        "type": "readout/pulse",
        "pulse_cfg": {
            "type": "pulse",
            "waveform": {"style": "const", "length": 0.5},
            "ch": 3,
            "nqz": 1,
            "freq": 7000.0,
            "gain": 0.8,
        },
        "ro_cfg": {"type": "readout/direct", "ro_ch": 0, "ro_length": 2.0},
    }
    d.update(kw)
    return d


def _make_pulse_cfg(ch=0, freq=5000.0):
    return PulseCfg(
        waveform=ConstWaveformCfg(length=0.5),
        ch=ch,
        nqz=1,
        freq=freq,
        gain=0.8,
    )


def _make_direct_cfg(ro_ch=0, ro_freq=5000.0):
    return DirectReadoutCfg(ro_ch=ro_ch, ro_length=1.0, ro_freq=ro_freq)


def _make_pulse_ro_cfg(ch=3, ro_ch=None):
    return PulseReadoutCfg(
        pulse_cfg=_make_pulse_cfg(ch=ch, freq=7000.0),
        ro_cfg=_make_direct_cfg(
            ro_ch=ro_ch if ro_ch is not None else ch, ro_freq=7000.0
        ),
    )


class TestDirectReadoutCfg:
    def test_validate_basic(self):
        cfg = DirectReadoutCfg.model_validate(_direct_dict())
        assert isinstance(cfg, DirectReadoutCfg)
        assert cfg.ro_ch == 0
        assert cfg.ro_length == 1.0
        assert cfg.ro_freq == 5000.0
        assert cfg.trig_offset == 0.0
        assert cfg.gen_ch is None

    def test_validate_with_optional_fields(self):
        cfg = DirectReadoutCfg.model_validate(_direct_dict(trig_offset=0.05, gen_ch=1))
        assert cfg.trig_offset == 0.05
        assert cfg.gen_ch == 1

    def test_module_adapter_dispatch(self):
        cfg = ModuleCfgFactory.from_raw(_direct_dict())
        assert isinstance(cfg, DirectReadoutCfg)
        assert cfg.trig_offset == 0.0
        assert cfg.gen_ch is None

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError, match="unknown_field"):
            DirectReadoutCfg.model_validate(_direct_dict(unknown_field=99))

    def test_set_param_ro_freq(self):
        cfg = _make_direct_cfg()
        cfg.set_param("ro_freq", 6000.0)
        assert cfg.ro_freq == 6000.0

    def test_set_param_ro_length(self):
        cfg = _make_direct_cfg()
        cfg.set_param("ro_length", 3.0)
        assert cfg.ro_length == 3.0

    def test_set_param_unknown_raises(self):
        cfg = _make_direct_cfg()
        with pytest.raises(ValueError, match="Unknown parameter: ro_ch"):
            cfg.set_param("ro_ch", 5)


class TestPulseReadoutCfg:
    def test_validate_basic(self):
        cfg = PulseReadoutCfg.model_validate(_pulse_dict())
        assert isinstance(cfg, PulseReadoutCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)
        assert isinstance(cfg.ro_cfg, DirectReadoutCfg)

    def test_auto_derives_ro_ch_from_pulse_ch(self):
        cfg = PulseReadoutCfg.model_validate(_pulse_dict())
        assert cfg.ro_cfg.gen_ch == 3  # derived from pulse_cfg.ch

    def test_auto_derives_ro_freq_from_pulse_freq(self):
        cfg = PulseReadoutCfg.model_validate(_pulse_dict())
        assert cfg.ro_cfg.ro_freq == 7000.0  # derived from pulse_cfg.freq

    def test_module_adapter_dispatch(self):
        cfg = ModuleCfgFactory.from_raw(_pulse_dict())
        assert isinstance(cfg, PulseReadoutCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)
        assert isinstance(cfg.ro_cfg, DirectReadoutCfg)

    def test_set_param_gain(self):
        cfg = _make_pulse_ro_cfg()
        cfg.set_param("gain", 0.3)
        assert cfg.pulse_cfg.gain == 0.3

    def test_set_param_freq_updates_both(self):
        cfg = _make_pulse_ro_cfg()
        cfg.set_param("freq", 8000.0)
        assert cfg.pulse_cfg.freq == 8000.0
        assert cfg.ro_cfg.ro_freq == 8000.0

    def test_set_param_ro_freq_only_updates_ro(self):
        cfg = _make_pulse_ro_cfg()
        original_pulse_freq = cfg.pulse_cfg.freq
        cfg.set_param("ro_freq", 8500.0)
        assert cfg.ro_cfg.ro_freq == 8500.0
        assert cfg.pulse_cfg.freq == original_pulse_freq

    def test_set_param_length_updates_waveform(self):
        cfg = _make_pulse_ro_cfg()
        cfg.set_param("length", 1.0)
        assert cfg.pulse_cfg.waveform.length == 1.0

    def test_set_param_ro_length(self):
        cfg = _make_pulse_ro_cfg()
        cfg.set_param("ro_length", 3.0)
        assert cfg.ro_cfg.ro_length == 3.0

    def test_set_param_unknown_raises(self):
        cfg = _make_pulse_ro_cfg()
        with pytest.raises(ValueError, match="Unknown parameter: ro_ch"):
            cfg.set_param("ro_ch", 5)


class TestReadoutCfgAdapter:
    def test_dispatch_direct(self):
        cfg = _READOUT_ADAPTER.validate_python(_direct_dict())
        assert isinstance(cfg, DirectReadoutCfg)

    def test_dispatch_pulse(self):
        cfg = _READOUT_ADAPTER.validate_python(_pulse_dict())
        assert isinstance(cfg, PulseReadoutCfg)
