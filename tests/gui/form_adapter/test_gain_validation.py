"""Gain validation reaches field-local errors before Form submission."""

from typing import Annotated

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import ConfigBase, FieldMeta, PulseSpec
from zcu_tools.gui.form import FormEnv, FormErrors, FormState
from zcu_tools.gui.form_adapter import Form, module_form
from zcu_tools.program.v2.sweep import GainSweepCfg, SweepCfg


class SweepPlan(ConfigBase):
    sweep: Annotated[GainSweepCfg, FieldMeta(unit="a.u.")]
    unrestricted: SweepCfg | None = None


class OptionalSweepPlan(ConfigBase):
    sweep: GainSweepCfg | None = None


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


@pytest.mark.parametrize(
    ("kind", "coordinates", "row"),
    [
        ("linrange", {"start": {"expr": "-1.25"}, "stop": 0, "n": 3}, "start"),
        ("linrange", {"start": 0, "stop": 1.25, "n": 3}, "stop"),
        ("linrange", {"start": 1.25, "stop": -1.25, "n": 3}, "start"),
        ("center", {"center": 1.1, "span": 0.6, "n": 3}, "center"),
        ("center", {"center": 0, "span": -2.5, "n": 3}, "span"),
        ("linrange_step", {"start": 1.25, "stop": 0, "step": -0.5}, "start"),
        ("linrange_step", {"start": 0, "stop": -1.5, "step": -0.6}, "step"),
        ("center_step", {"center": -0.95, "span": 0.6, "step": 0.4}, "span"),
        ("center_step", {"center": -1.1, "span": 0.6, "step": 0.4}, "center"),
        ("center_step", {"center": 0.75, "span": 1, "step": 0.4}, "step"),
        ("center_step", {"center": 0, "span": 3, "step": 0.7}, "span"),
    ],
    ids=[
        "range-start-expression",
        "range-stop",
        "range-both-first",
        "center-center",
        "center-span",
        "step-start",
        "step-final",
        "center-step-authored-center",
        "center-step-center",
        "center-step-final",
        "center-step-both-first",
    ],
)
def test_gain_sweep_errors_identify_one_visible_input_row(
    kind: str, coordinates: dict[str, JsonValue], row: str
) -> None:
    env = FormEnv()
    form = Form(SweepPlan, {})
    state = FormState(form.default_input(env))
    state.write({("sweep",): {"kind": kind, **coordinates}}, env)
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert list(result.messages) == [("sweep", row)]
    assert "[-1, 1]" in result.messages[("sweep", row)]


def test_gain_sweep_uses_effective_step_endpoints_and_leaves_plain_sweep_unbounded():
    env = FormEnv()
    form = Form(SweepPlan, {})
    state = FormState(form.default_input(env))
    state.write(
        {
            ("sweep",): {"kind": "linrange_step", "start": 0, "stop": 1.1, "step": 0.5},
            ("unrestricted",): {"kind": "linrange", "start": 0, "stop": 2, "n": 3},
        },
        env,
    )
    result = state.result(env)
    assert not isinstance(result, FormErrors)
    cfg = form.to_cfg(result, env)
    assert not isinstance(cfg, FormErrors)
    assert cfg.sweep.stop == 1.0
    assert cfg.unrestricted is not None
    assert cfg.unrestricted.stop == 2.0


def test_optional_gain_sweep_validates_only_the_selected_sweep_branch() -> None:
    env = FormEnv()
    form = Form(OptionalSweepPlan, {})
    state = FormState(form.default_input(env))
    result = state.result(env)
    assert not isinstance(result, FormErrors)
    cfg = form.to_cfg(result, env)
    assert not isinstance(cfg, FormErrors)
    assert cfg.sweep is None

    state.write({("sweep",): {"kind": "linrange", "start": 0, "stop": 2, "n": 3}}, env)
    invalid = state.result(env)
    assert isinstance(invalid, FormErrors)
    assert list(invalid.messages) == [("sweep", "stop")]
