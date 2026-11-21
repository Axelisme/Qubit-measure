"""Gain validation reaches field-local errors before Form submission."""

import pytest
from zcu_tools.cfg_model import PulseSpec
from zcu_tools.gui.form import FormEnv, FormErrors, FormState
from zcu_tools.gui.form_adapter import module_form


@pytest.mark.parametrize("value", [-1.0, 1.0, 1.1])
def test_fixed_gain_errors_stay_on_the_edited_field(value):
    env = FormEnv()
    form = module_form(PulseSpec)
    spec = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q.waveforms.flat")
    state = FormState(form.input_from_spec(spec, env))
    state.write({("gain",): value}, env)
    result = state.result(env)
    if value == 1.1:
        assert isinstance(result, FormErrors)
        assert "less than or equal to 1" in result.messages[("gain",)]
    else:
        assert not isinstance(result, FormErrors)
        cfg = form.to_cfg(result, env)
        assert not isinstance(cfg, FormErrors)
        assert cfg.gain == value
