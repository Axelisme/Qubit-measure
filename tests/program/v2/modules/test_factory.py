from unittest.mock import MagicMock

import pytest
from typing_extensions import Literal
from zcu_tools.program.v2.modules import (
    ModuleCfgFactory,
    WaveformCfgFactory,
)
from zcu_tools.program.v2.modules.base import ModuleCfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import (
    AbsReadoutCfg,
    DirectReadoutCfg,
    PulseReadoutCfg,
)
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    NoneResetCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import (
    ArbWaveformCfg,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
    WaveformCfg,
)

# ---------------------------------------------------------------------------
# ModuleCfgFactory
# ---------------------------------------------------------------------------


class TestModuleCfgFactoryRegistry:
    def test_all_leaf_cfgs_registered(self):
        expected = {
            "pulse": PulseCfg,
            "readout/direct": DirectReadoutCfg,
            "readout/pulse": PulseReadoutCfg,
            "reset/none": NoneResetCfg,
            "reset/pulse": PulseResetCfg,
            "reset/two_pulse": TwoPulseResetCfg,
            "reset/bath": BathResetCfg,
        }
        for type_value, cls in expected.items():
            assert ModuleCfgFactory._registry.get(type_value) is cls

    def test_registry_size_matches_known_leaves(self):
        assert len(ModuleCfgFactory._registry) == 7

    def test_abstract_intermediates_not_registered(self):
        assert AbsReadoutCfg not in ModuleCfgFactory._registry.values()


class TestModuleCfgFactoryFromRaw:
    def test_dispatch_pulse(self):
        cfg = ModuleCfgFactory.from_raw(
            {
                "type": "pulse",
                "waveform": {"style": "const", "length": 0.5},
                "ch": 1,
                "nqz": 1,
                "freq": 4000.0,
                "gain": 0.5,
            }
        )
        assert isinstance(cfg, PulseCfg)

    def test_dispatch_readout_direct(self):
        cfg = ModuleCfgFactory.from_raw(
            {"type": "readout/direct", "ro_ch": 0, "ro_length": 1.0, "ro_freq": 5000.0}
        )
        assert isinstance(cfg, DirectReadoutCfg)

    def test_dispatch_reset_none(self):
        cfg = ModuleCfgFactory.from_raw({"type": "reset/none"})
        assert isinstance(cfg, NoneResetCfg)

    def test_unknown_type_raises(self):
        with pytest.raises(Exception):
            ModuleCfgFactory.from_raw({"type": "no_such_module"})

    def test_missing_discriminator_raises(self):
        with pytest.raises(Exception):
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


class TestModuleCfgFactoryRegister:
    def test_idempotent_register_same_class(self):
        before = len(ModuleCfgFactory._registry)
        ModuleCfgFactory.register(PulseCfg)
        assert len(ModuleCfgFactory._registry) == before

    def test_reject_non_literal_discriminator(self):
        class BadCfg(ModuleCfg):
            type: str = "bad"  # not a Literal

        with pytest.raises(TypeError):
            ModuleCfgFactory.register(BadCfg)

    def test_reject_conflicting_discriminator(self):
        class DuplicatePulseCfg(ModuleCfg):
            type: Literal["pulse"] = "pulse"  # collides with PulseCfg

        with pytest.raises(ValueError):
            ModuleCfgFactory.register(DuplicatePulseCfg)


# ---------------------------------------------------------------------------
# WaveformCfgFactory
# ---------------------------------------------------------------------------


class TestWaveformCfgFactoryRegistry:
    def test_all_leaf_cfgs_registered(self):
        expected = {
            "const": ConstWaveformCfg,
            "cosine": CosineWaveformCfg,
            "gauss": GaussWaveformCfg,
            "drag": DragWaveformCfg,
            "arb": ArbWaveformCfg,
            "flat_top": FlatTopWaveformCfg,
        }
        for style_value, cls in expected.items():
            assert WaveformCfgFactory._registry.get(style_value) is cls

    def test_registry_size_matches_known_leaves(self):
        assert len(WaveformCfgFactory._registry) == 6


class TestWaveformCfgFactoryFromRaw:
    def test_dispatch_const(self):
        cfg = WaveformCfgFactory.from_raw({"style": "const", "length": 1.0})
        assert isinstance(cfg, ConstWaveformCfg)

    def test_dispatch_gauss(self):
        cfg = WaveformCfgFactory.from_raw(
            {"style": "gauss", "length": 1.0, "sigma": 0.2}
        )
        assert isinstance(cfg, GaussWaveformCfg)
        assert cfg.sigma == 0.2

    def test_dispatch_flat_top_with_nested_raise(self):
        cfg = WaveformCfgFactory.from_raw(
            {
                "style": "flat_top",
                "length": 1.0,
                "raise_waveform": {"style": "gauss", "length": 0.2, "sigma": 0.05},
            }
        )
        assert isinstance(cfg, FlatTopWaveformCfg)
        assert isinstance(cfg.raise_waveform, GaussWaveformCfg)

    def test_unknown_style_raises(self):
        with pytest.raises(Exception):
            WaveformCfgFactory.from_raw({"style": "no_such_style", "length": 1.0})

    def test_missing_discriminator_raises(self):
        with pytest.raises(Exception):
            WaveformCfgFactory.from_raw({"length": 1.0})


class TestWaveformCfgFactoryRegister:
    def test_idempotent_register_same_class(self):
        before = len(WaveformCfgFactory._registry)
        WaveformCfgFactory.register(ConstWaveformCfg)
        assert len(WaveformCfgFactory._registry) == before

    def test_reject_non_literal_discriminator(self):
        class BadWaveformCfg(WaveformCfg):
            style: str = "bad"  # not a Literal

        with pytest.raises(TypeError):
            WaveformCfgFactory.register(BadWaveformCfg)

    def test_reject_conflicting_discriminator(self):
        class DuplicateConstCfg(WaveformCfg):
            style: Literal["const"] = "const"  # collides with ConstWaveformCfg

        with pytest.raises(ValueError):
            WaveformCfgFactory.register(DuplicateConstCfg)
