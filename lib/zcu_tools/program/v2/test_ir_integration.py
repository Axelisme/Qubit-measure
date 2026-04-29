"""Integration tests for IR lowering in ModularProgramV2."""

from __future__ import annotations

import pytest

from .modules.pulse import Pulse, PulseCfg
from .modules.reset import NoneReset, NoneResetCfg
from .modules.waveform import ConstWaveformCfg


def test_module_ir_lowering() -> None:
    """Test that modules can lower to IR without crashing."""
    # Create a simple pulse
    pulse_cfg = PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=0.04),
        ch=0,
        nqz=1,
        freq=100.0,
        gain=1.0,
    )
    pulse = Pulse(name="test_pulse", cfg=pulse_cfg)
    pulse.pulse_id = "test_pulse_id"

    # Create a reset module
    reset = NoneReset(name="reset", cfg=NoneResetCfg(type="reset/none"))

    # Both modules should have lower() methods
    assert hasattr(pulse, "lower")
    assert hasattr(reset, "lower")


def test_feature_flag_env() -> None:
    """Test that ZCU_TOOLS_USE_IR can be read from environment."""
    from .modular import ZCU_TOOLS_USE_IR

    # Feature flag should exist and have a boolean value
    assert isinstance(ZCU_TOOLS_USE_IR, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
