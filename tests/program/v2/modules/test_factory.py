from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from zcu_tools.program.v2.modules import ModuleCfgFactory, WaveformCfgFactory
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.reset import NoneResetCfg, PulseResetCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


class TestModuleCfgFactoryFromRaw:
    def test_unknown_type_raises(self):
        with pytest.raises(ValidationError, match="union_tag_invalid"):
            ModuleCfgFactory.from_raw({"type": "no_such_module"})

    def test_missing_discriminator_raises(self):
        with pytest.raises(ValidationError, match="union_tag_not_found"):
            ModuleCfgFactory.from_raw({"ch": 1})

    def test_ml_context_propagated_for_string_ref(self):
        ml = MagicMock()
        ml.get_module.return_value = PulseCfg(
            waveform=ConstWaveformCfg(length=0.2),
            ch=1,
            nqz=1,
            freq=4000.0,
            gain=0.5,
        )
        cfg = ModuleCfgFactory.from_raw(
            {"type": "reset/pulse", "pulse_cfg": "library_pulse"}, ml=ml
        )
        assert isinstance(cfg, PulseResetCfg)
        ml.get_module.assert_called_once_with("library_pulse")

    def test_no_ml_context_when_not_provided(self):
        cfg = ModuleCfgFactory.from_raw({"type": "reset/none"})
        assert isinstance(cfg, NoneResetCfg)


class TestWaveformCfgFactoryFromRaw:
    def test_unknown_style_raises(self):
        with pytest.raises(ValidationError, match="union_tag_invalid"):
            WaveformCfgFactory.from_raw({"style": "no_such_style", "length": 1.0})

    def test_missing_discriminator_raises(self):
        with pytest.raises(ValidationError, match="union_tag_not_found"):
            WaveformCfgFactory.from_raw({"length": 1.0})
