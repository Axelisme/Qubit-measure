from pydantic import BaseModel
from zcu_tools.program.v2.modules.pulse import Pulse


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


def test_pulse_none_cfg_is_no_op(mock_prog):
    p = Pulse("p", None)
    assert p.run(mock_prog, t=0.5) == 0.5
    assert p.total_length(mock_prog) == 0.0
    assert p.allow_rerun() is True
