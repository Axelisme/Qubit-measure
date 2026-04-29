from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter
from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.modules import (
    ModuleCfgFactory,  # ensures leaf subclass registration
)
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import (
    AbsReadoutCfg,
    DirectReadout,
    DirectReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
    Readout,
    ReadoutCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_READOUT_ADAPTER: TypeAdapter = TypeAdapter(ReadoutCfg)


def _ml():
    return MagicMock()


def _direct_dict(**kw):
    base = {"type": "readout/direct", "ro_ch": 0, "ro_length": 1.0, "ro_freq": 5000.0}
    base.update(kw)
    return base


def _pulse_dict(**kw):
    d = {
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


# ---------------------------------------------------------------------------
# DirectReadoutCfg – validation / set_param
# ---------------------------------------------------------------------------


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
        with pytest.raises(Exception):
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
        with pytest.raises(ValueError):
            cfg.set_param("ro_ch", 5)


# ---------------------------------------------------------------------------
# PulseReadoutCfg – validation / auto-derivation / set_param
# ---------------------------------------------------------------------------


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

    def test_explicit_ro_ch_not_overwritten(self):
        d = _pulse_dict()
        d["ro_cfg"]["ro_ch"] = 9  # type: ignore[index]
        cfg = PulseReadoutCfg.model_validate(d)
        assert cfg.ro_cfg.ro_ch == 9

    def test_explicit_ro_freq_not_overwritten(self):
        d = _pulse_dict()
        d["ro_cfg"]["ro_ch"] = 3  # type: ignore[index]
        d["ro_cfg"]["ro_freq"] = 9999.0  # type: ignore[index]
        cfg = PulseReadoutCfg.model_validate(d)
        assert cfg.ro_cfg.ro_freq == 9999.0

    def test_pulse_cfg_as_string_uses_ml_context(self):
        ml = _ml()
        ml.get_module.return_value = _make_pulse_cfg(ch=0, freq=5000.0)
        d = {
            "type": "readout/pulse",
            "pulse_cfg": "library_pulse",
            "ro_cfg": {"ro_ch": 0, "ro_length": 1.0, "ro_freq": 5000.0},
        }
        cfg = PulseReadoutCfg.model_validate(d, context={"ml": ml})
        ml.get_module.assert_called_once_with("library_pulse")
        assert isinstance(cfg, PulseReadoutCfg)
        assert isinstance(cfg.pulse_cfg, PulseCfg)

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
        with pytest.raises(ValueError):
            cfg.set_param("ro_ch", 5)


# ---------------------------------------------------------------------------
# Readout factory – dispatch
# ---------------------------------------------------------------------------


class TestReadoutFactory:
    def test_dispatch_direct(self):
        ro = Readout("ro", _make_direct_cfg())
        assert isinstance(ro, DirectReadout)

    def test_dispatch_pulse(self):
        ro = Readout("ro", _make_pulse_ro_cfg())
        assert isinstance(ro, PulseReadout)

    def test_unknown_cfg_raises(self):
        class UnknownCfg(AbsReadoutCfg):
            type: str = "unknown"

            def build(self, name):
                raise NotImplementedError

        with pytest.raises(NotImplementedError):
            Readout("ro", UnknownCfg())

    def test_name_set(self):
        ro = Readout("myro", _make_direct_cfg())
        assert ro.name == "myro"


# ---------------------------------------------------------------------------
# DirectReadout / PulseReadout – init / run via mock prog
# ---------------------------------------------------------------------------


class TestDirectReadoutRuntime:
    def test_init_calls_declare_and_add(self, mock_prog):
        ro = DirectReadout("ro", _make_direct_cfg(ro_ch=2, ro_freq=6000.0))
        ro.init(mock_prog)
        mock_prog.declare_readout.assert_called_once_with(ch=2, length=1.0)
        mock_prog.add_readoutconfig.assert_called_once()

    def test_init_passes_gen_ch_when_set(self, mock_prog):
        cfg = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=5000.0, gen_ch=1)
        ro = DirectReadout("ro", cfg)
        ro.init(mock_prog)
        _, kwargs = mock_prog.add_readoutconfig.call_args
        assert kwargs.get("gen_ch") == 1

    def test_init_omits_gen_ch_when_none(self, mock_prog):
        ro = DirectReadout("ro", _make_direct_cfg())
        ro.init(mock_prog)
        _, kwargs = mock_prog.add_readoutconfig.call_args
        assert "gen_ch" not in kwargs

    def test_ir_run_calls_send_readoutconfig_and_trigger(self, mock_prog):
        ro = DirectReadout("ro", _make_direct_cfg())
        ro.init(mock_prog)
        b = IRBuilder()
        result = ro.ir_run(b, t=0.5, prog=mock_prog)
        root = b.build()
        assert hasattr(root, "meta")
        assert result == 0.5  # returns t unchanged



class TestPulseReadoutRuntime:
    def test_init_wires_pulse_and_ro_window(self, mock_prog):
        ro = PulseReadout("ro", _make_pulse_ro_cfg(ch=3))
        ro.init(mock_prog)
        mock_prog.declare_readout.assert_called_once()
        mock_prog.declare_gen.assert_called_once()

    def test_ir_run_calls_both_submodules(self, mock_prog):
        ro = PulseReadout("ro", _make_pulse_ro_cfg(ch=3))
        ro.init(mock_prog)
        b = IRBuilder()
        out = ro.ir_run(b, t=0.0, prog=mock_prog)
        assert out == 0.0



# ---------------------------------------------------------------------------
# ReadoutCfg discriminated union adapter
# ---------------------------------------------------------------------------


class TestReadoutCfgAdapter:
    def test_dispatch_direct(self):
        cfg = _READOUT_ADAPTER.validate_python(_direct_dict())
        assert isinstance(cfg, DirectReadoutCfg)

    def test_dispatch_pulse(self):
        cfg = _READOUT_ADAPTER.validate_python(_pulse_dict())
        assert isinstance(cfg, PulseReadoutCfg)
