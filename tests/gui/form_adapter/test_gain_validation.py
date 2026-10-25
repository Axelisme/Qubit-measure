"""Gain validation reaches field-local errors before Form submission."""

from typing import Annotated

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import ConfigBase, FieldMeta, Gain, PulseSpec, Range
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormState
from zcu_tools.gui.form_adapter import Form, module_form
from zcu_tools.program.v2 import GainSweep


class SyntheticOptionalGainCfg(ConfigBase):
    scan: GainSweep | None = None


class SyntheticGainCfg(ConfigBase):
    amplitude: Annotated[Gain, FieldMeta(unit="a.u.")]
    scan: Annotated[GainSweep, FieldMeta(unit="a.u.")]


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
        assert "Gain must be within [-1, 1]" in result.messages[("gain",)]
        assert "1.1" in result.messages[("gain",)]
    else:
        assert not isinstance(result, FormErrors)
        cfg = form.to_cfg(result, env)
        assert not isinstance(cfg, FormErrors)
        assert cfg.gain == value


@pytest.mark.parametrize(
    "raw,invalid",
    [
        ({"kind": "linrange", "start": -1.0, "stop": 1.0, "n": 3}, False),
        ({"kind": "linrange", "start": -1.0, "stop": 1.1, "n": 3}, True),
        ({"kind": "linrange", "start": -1.1, "stop": 1.0, "n": 3}, True),
        ({"kind": "center", "center": 0.0, "span": 2.2, "n": 3}, True),
        ({"kind": "linrange_step", "start": -1.0, "stop": 1.1, "step": 0.5}, False),
        ({"kind": "center_step", "center": 0.05, "span": 2.1, "step": 0.5}, False),
        ({"kind": "center_step", "center": 0.0, "span": 2.2, "step": 0.5}, True),
    ],
)
def test_gain_sweep_errors_stay_on_the_declared_field(raw, invalid):
    env = FormEnv()
    form = Form(SyntheticGainCfg, {"amplitude": 0.2, "scan": Range(-1, 1, 3)})
    state = FormState(FormInput(form.model, {"amplitude": 0.2, "scan": raw}))
    result = state.result(env)
    if invalid:
        assert isinstance(result, FormErrors)
        assert "Gain must be within [-1, 1]" in result.messages[("scan",)]
        assert "1.1" in result.messages[("scan",)]
    else:
        assert not isinstance(result, FormErrors)
        cfg = form.to_cfg(result, env)
        assert not isinstance(cfg, FormErrors)
        assert (cfg.scan.start, cfg.scan.stop) == (-1.0, 1.0)


def test_nullable_sweep_preserves_its_branch_validation():
    env = FormEnv()
    form = Form(SyntheticOptionalGainCfg, {})
    inputs: tuple[JsonValue, ...] = (
        None,
        {"kind": "linrange", "start": -1.0, "stop": 1.0, "n": 3},
    )
    for raw in inputs:
        result = FormState(FormInput(form.model, {"scan": raw})).result(env)
        assert not isinstance(result, FormErrors)
    state = FormState(
        FormInput(
            form.model,
            {"scan": {"kind": "linrange", "start": -1.0, "stop": 1.1, "n": 3}},
        )
    )
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert "Gain must be within [-1, 1]" in result.messages[("scan",)]
