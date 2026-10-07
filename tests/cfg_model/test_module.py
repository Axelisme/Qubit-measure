"""Core module validation and runtime knob contracts."""

from copy import deepcopy

import pytest
from pydantic import TypeAdapter, ValidationError
from zcu_tools.cfg_model import (
    BathResetCfg,
    CfgInputMap,
    CfgInputValue,
    CfgValidationContext,
    ConstWaveformCfg,
    DirectReadoutCfg,
    GaussWaveformCfg,
    ModuleCfgFactory,
    NoneResetCfg,
    PulseCfg,
    PulseReadoutCfg,
    PulseResetCfg,
    PulseSpec,
    ReadoutCfg,
    ResetCfg,
    TwoPulseResetCfg,
)

from ._support import Resolver

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
    def test_explicit_ro_ch_not_overwritten(self):
        d = _pulse_dict(ro_cfg={"type": "readout/direct", "ro_ch": 9, "ro_length": 2.0})
        cfg = PulseReadoutCfg.model_validate(d)
        assert cfg.ro_cfg.ro_ch == 9

    def test_explicit_ro_freq_not_overwritten(self):
        d = _pulse_dict(
            ro_cfg={
                "type": "readout/direct",
                "ro_ch": 3,
                "ro_length": 2.0,
                "ro_freq": 9999.0,
            }
        )
        cfg = PulseReadoutCfg.model_validate(d)
        assert cfg.ro_cfg.ro_freq == 9999.0

    def test_pulse_cfg_as_string_uses_typed_context(self):
        resolver = Resolver()
        pulse = PulseSpec(
            ch=0,
            nqz=1,
            freq=5000.0,
            gain=0.8,
            waveform=ConstWaveformCfg(length=0.5),
        )
        resolver.modules["library_pulse"] = pulse
        d: CfgInputMap = {
            "type": "readout/pulse",
            "pulse_cfg": "library_pulse",
            "ro_cfg": {"ro_ch": 0, "ro_length": 1.0, "ro_freq": 5000.0},
        }
        input_before = deepcopy(d)
        spec_before = pulse.model_dump()

        cfg = ModuleCfgFactory.from_raw(
            d, context=CfgValidationContext(modules=resolver)
        )

        assert isinstance(cfg, PulseReadoutCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)
        assert cfg.pulse_cfg.ch == 0
        assert cfg.pulse_cfg.freq == 5000.0
        assert cfg.pulse_cfg.gain == 0.8
        assert cfg.pulse_cfg.waveform.length == 0.5
        assert cfg.ro_cfg.ro_ch == 0
        assert cfg.ro_cfg.ro_length == 1.0
        assert cfg.ro_cfg.ro_freq == 5000.0
        cfg.pulse_cfg.set_param("gain", 0.2)
        cfg.set_param("ro_freq", 6000.0)
        assert d == input_before
        assert pulse.model_dump() == spec_before

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


_RESET_ADAPTER: TypeAdapter[ResetCfg] = TypeAdapter(ResetCfg)


def _reset_pulse_dict(ch=1, freq=4000.0, length=0.2, gain=0.5) -> CfgInputMap:
    return {
        "type": "pulse",
        "waveform": {"style": "const", "length": length},
        "ch": ch,
        "nqz": 1,
        "freq": freq,
        "gain": gain,
    }


def _make_reset_pulse_cfg(ch=1, freq=4000.0, length=0.2, gain=0.5):
    return PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=ch,
        nqz=1,
        freq=freq,
        gain=gain,
    )


def _make_reset_pi2_cfg():
    return PulseCfg(
        waveform=GaussWaveformCfg(length=0.1, sigma=0.025),
        ch=2,
        nqz=1,
        freq=4000.0,
        gain=0.5,
    )


class TestNoneResetCfg:
    def test_validate_basic(self):
        cfg = NoneResetCfg.model_validate({"type": "reset/none"})
        assert isinstance(cfg, NoneResetCfg)

    def test_module_adapter_dispatch(self):
        cfg = ModuleCfgFactory.from_raw({"type": "reset/none"})
        assert isinstance(cfg, NoneResetCfg)

    def test_set_param_always_raises(self):
        cfg = NoneResetCfg()
        with pytest.raises(ValueError, match="NoneReset does not support set_param"):
            cfg.set_param("anything", 1.0)


class TestPulseResetCfg:
    def test_pulse_cfg_as_string_uses_typed_context(self):
        resolver = Resolver()
        pulse = PulseSpec(
            ch=1,
            nqz=1,
            freq=4000.0,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.2),
        )
        resolver.modules["my_pulse"] = pulse
        raw: CfgInputMap = {"type": "reset/pulse", "pulse_cfg": "my_pulse"}
        input_before = deepcopy(raw)
        spec_before = pulse.model_dump()

        cfg = ModuleCfgFactory.from_raw(
            raw, context=CfgValidationContext(modules=resolver)
        )

        assert isinstance(cfg, PulseResetCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)
        assert cfg.pulse_cfg.ch == 1
        assert cfg.pulse_cfg.freq == 4000.0
        assert cfg.pulse_cfg.gain == 0.5
        assert cfg.pulse_cfg.waveform.length == 0.2
        cfg.set_param("gain", 0.9)
        cfg.set_param("length", 0.5)
        assert raw == input_before
        assert pulse.model_dump() == spec_before

    def _dict(self) -> CfgInputMap:
        return {"type": "reset/pulse", "pulse_cfg": _reset_pulse_dict()}

    def test_validate_basic(self):
        cfg = PulseResetCfg.model_validate(self._dict())
        assert isinstance(cfg, PulseResetCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)
        assert cfg.pulse_cfg.ch == 1

    def test_validate_pulse_cfg_freq(self):
        cfg = PulseResetCfg.model_validate(self._dict())
        assert cfg.pulse_cfg.freq == 4000.0

    def test_module_adapter_dispatch(self):
        cfg = ModuleCfgFactory.from_raw(self._dict())
        assert isinstance(cfg, PulseResetCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)

    def test_set_param_gain(self):
        cfg = PulseResetCfg(pulse_cfg=_make_reset_pulse_cfg())
        cfg.set_param("gain", 0.9)
        assert cfg.pulse_cfg.gain == 0.9

    def test_set_param_freq(self):
        cfg = PulseResetCfg(pulse_cfg=_make_reset_pulse_cfg())
        cfg.set_param("freq", 5000.0)
        assert cfg.pulse_cfg.freq == 5000.0

    def test_set_param_length(self):
        cfg = PulseResetCfg(pulse_cfg=_make_reset_pulse_cfg(length=0.2))
        cfg.set_param("length", 0.5)
        assert cfg.pulse_cfg.waveform.length == 0.5

    def test_set_param_unknown_raises(self):
        cfg = PulseResetCfg(pulse_cfg=_make_reset_pulse_cfg())
        with pytest.raises(ValueError, match="Unknown parameter: ro_ch"):
            cfg.set_param("ro_ch", 0)


class TestTwoPulseResetCfg:
    def test_pulse_cfgs_as_strings_use_typed_context(self):
        resolver = Resolver()
        pulse = PulseSpec(
            ch=1,
            nqz=1,
            freq=4000.0,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.2),
        )
        resolver.modules["p1"] = pulse
        resolver.modules["p2"] = pulse
        raw: CfgInputMap = {
            "type": "reset/two_pulse",
            "pulse1_cfg": "p1",
            "pulse2_cfg": "p2",
        }
        input_before = deepcopy(raw)
        spec_before = pulse.model_dump()

        cfg = ModuleCfgFactory.from_raw(
            raw, context=CfgValidationContext(modules=resolver)
        )

        assert isinstance(cfg, TwoPulseResetCfg)
        for child in (cfg.pulse1_cfg, cfg.pulse2_cfg):
            assert isinstance(child, PulseCfg)
            assert child.ch == 1
            assert child.freq == 4000.0
            assert child.gain == 0.5
            assert child.waveform.length == 0.2
        cfg.set_param("gain1", 0.9)
        cfg.pulse1_cfg.set_param("length", 0.5)
        assert cfg.pulse2_cfg.gain == 0.5
        assert cfg.pulse2_cfg.waveform.length == 0.2
        assert raw == input_before
        assert pulse.model_dump() == spec_before

    def _dict(self) -> CfgInputMap:
        return {
            "type": "reset/two_pulse",
            "pulse1_cfg": _reset_pulse_dict(ch=1, freq=4000.0, length=0.2),
            "pulse2_cfg": _reset_pulse_dict(ch=2, freq=3000.0, length=0.3),
        }

    def test_validate_basic(self):
        cfg = TwoPulseResetCfg.model_validate(self._dict())
        assert isinstance(cfg, TwoPulseResetCfg)
        assert isinstance(cfg.pulse1_cfg, PulseCfg)
        assert isinstance(cfg.pulse2_cfg, PulseCfg)

    def test_preserves_individual_channels(self):
        cfg = TwoPulseResetCfg.model_validate(self._dict())
        assert cfg.pulse1_cfg.ch == 1
        assert cfg.pulse2_cfg.ch == 2

    def test_module_adapter_dispatch(self):
        cfg = ModuleCfgFactory.from_raw(self._dict())
        assert isinstance(cfg, TwoPulseResetCfg)
        assert isinstance(cfg.pulse1_cfg, PulseCfg)
        assert isinstance(cfg.pulse2_cfg, PulseCfg)

    def test_set_param_gain1(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_reset_pulse_cfg(), pulse2_cfg=_make_reset_pulse_cfg(ch=2)
        )
        cfg.set_param("gain1", 0.1)
        assert cfg.pulse1_cfg.gain == 0.1

    def test_set_param_gain2(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_reset_pulse_cfg(), pulse2_cfg=_make_reset_pulse_cfg(ch=2)
        )
        cfg.set_param("gain2", 0.2)
        assert cfg.pulse2_cfg.gain == 0.2

    def test_set_param_freq1(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_reset_pulse_cfg(), pulse2_cfg=_make_reset_pulse_cfg(ch=2)
        )
        cfg.set_param("freq1", 5000.0)
        assert cfg.pulse1_cfg.freq == 5000.0

    def test_set_param_freq2(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_reset_pulse_cfg(), pulse2_cfg=_make_reset_pulse_cfg(ch=2)
        )
        cfg.set_param("freq2", 5000.0)
        assert cfg.pulse2_cfg.freq == 5000.0

    def test_set_param_length_sets_both(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_reset_pulse_cfg(length=0.2),
            pulse2_cfg=_make_reset_pulse_cfg(ch=2, length=0.3),
        )
        cfg.set_param("length", 0.5)
        assert cfg.pulse1_cfg.waveform.length == 0.5
        assert cfg.pulse2_cfg.waveform.length == 0.5

    def test_set_param_unknown_raises(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_make_reset_pulse_cfg(), pulse2_cfg=_make_reset_pulse_cfg(ch=2)
        )
        with pytest.raises(ValueError, match="Unknown parameter: unknown"):
            cfg.set_param("unknown", 0)


class TestBathResetCfg:
    def test_cfgs_as_strings_use_typed_context(self):
        resolver = Resolver()
        pulse = PulseSpec(
            ch=1,
            nqz=1,
            freq=4000.0,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.2),
        )
        resolver.modules["res_pulse"] = pulse
        resolver.modules["qub_pulse"] = pulse
        resolver.modules["pi2_pulse"] = pulse
        raw: CfgInputMap = {
            "type": "reset/bath",
            "cavity_tone_cfg": "res_pulse",
            "qubit_tone_cfg": "qub_pulse",
            "pi2_cfg": "pi2_pulse",
        }
        input_before = deepcopy(raw)
        spec_before = pulse.model_dump()

        cfg = ModuleCfgFactory.from_raw(
            raw, context=CfgValidationContext(modules=resolver)
        )

        assert isinstance(cfg, BathResetCfg)
        for child in (cfg.cavity_tone_cfg, cfg.qubit_tone_cfg, cfg.pi2_cfg):
            assert isinstance(child, PulseCfg)
            assert child.ch == 1
            assert child.freq == 4000.0
            assert child.gain == 0.5
            assert child.waveform.length == 0.2
        cfg.set_param("res_gain", 0.9)
        cfg.cavity_tone_cfg.set_param("length", 0.5)
        assert cfg.qubit_tone_cfg.gain == 0.5
        assert cfg.qubit_tone_cfg.waveform.length == 0.2
        assert cfg.pi2_cfg.gain == 0.5
        assert cfg.pi2_cfg.waveform.length == 0.2
        assert raw == input_before
        assert pulse.model_dump() == spec_before

    def _dict(self) -> CfgInputMap:
        return {
            "type": "reset/bath",
            "cavity_tone_cfg": _reset_pulse_dict(ch=1, freq=5000.0, length=5.0),
            "qubit_tone_cfg": _reset_pulse_dict(ch=2, freq=4000.0, length=5.0),
            "pi2_cfg": {
                "type": "pulse",
                "waveform": {"style": "gauss", "length": 0.1, "sigma": 0.025},
                "ch": 2,
                "nqz": 1,
                "freq": 4000.0,
                "gain": 0.5,
            },
        }

    def test_validate_basic(self):
        cfg = BathResetCfg.model_validate(self._dict())
        assert isinstance(cfg, BathResetCfg)
        assert isinstance(cfg.cavity_tone_cfg, PulseCfg)
        assert isinstance(cfg.qubit_tone_cfg, PulseCfg)
        assert isinstance(cfg.pi2_cfg, PulseCfg)

    def test_module_adapter_dispatch(self):
        cfg = ModuleCfgFactory.from_raw(self._dict())
        assert isinstance(cfg, BathResetCfg)

    def test_set_param_qub_gain(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_reset_pulse_cfg(ch=1),
            qubit_tone_cfg=_make_reset_pulse_cfg(ch=2),
            pi2_cfg=_make_reset_pi2_cfg(),
        )
        cfg.set_param("qub_gain", 0.1)
        assert cfg.qubit_tone_cfg.gain == 0.1

    def test_set_param_qub_freq(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_reset_pulse_cfg(ch=1),
            qubit_tone_cfg=_make_reset_pulse_cfg(ch=2, freq=4000.0),
            pi2_cfg=_make_reset_pi2_cfg(),
        )
        cfg.set_param("qub_freq", 4500.0)
        assert cfg.qubit_tone_cfg.freq == 4500.0

    def test_set_param_res_gain(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_reset_pulse_cfg(ch=1),
            qubit_tone_cfg=_make_reset_pulse_cfg(ch=2),
            pi2_cfg=_make_reset_pi2_cfg(),
        )
        cfg.set_param("res_gain", 0.9)
        assert cfg.cavity_tone_cfg.gain == 0.9

    def test_set_param_res_freq(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_reset_pulse_cfg(ch=1, freq=5000.0),
            qubit_tone_cfg=_make_reset_pulse_cfg(ch=2),
            pi2_cfg=_make_reset_pi2_cfg(),
        )
        cfg.set_param("res_freq", 5500.0)
        assert cfg.cavity_tone_cfg.freq == 5500.0

    def test_set_param_qub_length(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_reset_pulse_cfg(ch=1, length=5.0),
            qubit_tone_cfg=_make_reset_pulse_cfg(ch=2, length=5.0),
            pi2_cfg=_make_reset_pi2_cfg(),
        )
        cfg.set_param("qub_length", 3.0)
        assert cfg.qubit_tone_cfg.waveform.length == 3.0

    def test_set_param_res_length(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_reset_pulse_cfg(ch=1, length=5.0),
            qubit_tone_cfg=_make_reset_pulse_cfg(ch=2, length=5.0),
            pi2_cfg=_make_reset_pi2_cfg(),
        )
        cfg.set_param("res_length", 7.0)
        assert cfg.cavity_tone_cfg.waveform.length == 7.0

    def test_set_param_pi2_phase(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_reset_pulse_cfg(ch=1),
            qubit_tone_cfg=_make_reset_pulse_cfg(ch=2),
            pi2_cfg=_make_reset_pi2_cfg(),
        )
        cfg.set_param("pi2_phase", 90.0)
        assert cfg.pi2_cfg.phase == 90.0

    def test_set_param_unknown_raises(self):
        cfg = BathResetCfg(
            cavity_tone_cfg=_make_reset_pulse_cfg(ch=1),
            qubit_tone_cfg=_make_reset_pulse_cfg(ch=2),
            pi2_cfg=_make_reset_pi2_cfg(),
        )
        with pytest.raises(ValueError, match="Unknown parameter: unknown"):
            cfg.set_param("unknown", 0)


class TestResetCfgAdapter:
    def test_dispatch_none(self):
        cfg = _RESET_ADAPTER.validate_python({"type": "reset/none"})
        assert isinstance(cfg, NoneResetCfg)

    def test_dispatch_pulse(self):
        cfg = _RESET_ADAPTER.validate_python(
            {"type": "reset/pulse", "pulse_cfg": _reset_pulse_dict()}
        )
        assert isinstance(cfg, PulseResetCfg)
