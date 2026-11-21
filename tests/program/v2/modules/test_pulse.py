from math import inf, nextafter

import pytest
from pydantic import BaseModel, ValidationError
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def test_pulse_cfg_namespace_validates_nested_input():
    class Modules(BaseModel):
        pulse: Pulse.Cfg

    modules = Modules.model_validate(
        {
            "pulse": {
                "ch": 0,
                "nqz": 1,
                "freq": 500.0,
                "gain": 0.2,
                "waveform": {"style": "const", "length": 1.0},
            }
        }
    )
    assert modules.pulse.freq == 500.0
    assert modules.pulse.waveform.length == 1.0
    assert Pulse("drive", modules.pulse).cfg == modules.pulse


@pytest.mark.parametrize("gain", [1.2, -1.2, nextafter(1.0, inf)])
def test_scalar_gain_outside_full_scale_fails_at_construction(gain: float) -> None:
    with pytest.raises(ValidationError) as error:
        PulseCfg(
            ch=0,
            nqz=1,
            freq=100.0,
            gain=gain,
            waveform=ConstWaveformCfg(length=0.1),
        )

    assert {item["loc"][0] for item in error.value.errors()} == {"gain"}
