"""Sweep coordinate commands preserve drafts and lower to the same native cfg."""

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import ConfigBase, Expr, Range
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPathError,
    FormState,
)
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2.sweep import SweepCfg


class CoordinateCfg(ConfigBase):
    sweep: SweepCfg
    other: float = 1


def switch(state: FormState[FormModel], env: FormEnv, field: str, label: str) -> None:
    target = state.describe(("sweep", field), env).target
    commands = getattr(target, "commands", ())
    command = next((item for item in commands if item.label == label), None)
    assert command is not None, f"Missing coordinate command {label}"
    state.run_command(("sweep", field), command.name, env)


def test_descending_coordinate_roundtrip_and_lowering():
    env = FormEnv()
    form = Form(CoordinateCfg, {"sweep": Range.between(10, 4, 4)})
    state = FormState(form.default_input(env))
    initial = state.result(env)
    assert not isinstance(initial, FormErrors)
    original = form.to_cfg(initial, env)
    state.write(
        {("sweep", "start"): {"expr": "5 * 2"}, ("sweep", "n"): {"expr": "2 * 2"}}, env
    )
    switch(state, env, "start", "Switch to Center / Span")
    assert state.input.data["sweep"] == {
        "kind": "center",
        "center": 7.0,
        "span": -6.0,
        "n": {"expr": "2 * 2"},
    }
    assert [
        item.name
        for item in state.describe(("sweep",), env).children
        if not item.hidden
    ] == ["center", "span", "n"]
    assert state.preview(env).values[("sweep", "span")] == -6
    with pytest.raises(FormPathError):
        state.describe(("sweep", "start"), env)
    result = state.result(env)
    assert not isinstance(result, FormErrors)
    assert form.to_cfg(result, env) == original
    switch(state, env, "span", "Switch to Range")
    assert state.input.data["sweep"] == {
        "kind": "linrange",
        "start": 10.0,
        "stop": 4.0,
        "n": {"expr": "2 * 2"},
    }


@pytest.mark.parametrize("sampling", [None, {"expr": "1 / 0"}, {"expr": "2 + 2"}])
def test_coordinate_command_ignores_unrelated_and_sampling_errors(sampling: JsonValue):
    env = FormEnv()
    form = Form(CoordinateCfg, {"sweep": Range.between(0, 10, 4)})
    raw: dict[str, JsonValue] = {"kind": "linrange", "start": 0, "stop": 10}
    if sampling is not None:
        raw["n"] = sampling
    state = FormState(FormInput(form.model, {"sweep": raw, "other": {"expr": "1 / 0"}}))
    switch(state, env, "stop", "Switch to Center / Span")
    expected: dict[str, JsonValue] = {"kind": "center", "center": 5.0, "span": 10.0}
    if sampling is not None:
        expected["n"] = sampling
    assert state.input.data == {"sweep": expected, "other": {"expr": "1 / 0"}}


@pytest.mark.parametrize("endpoint", [None, {"expr": "1 / 0"}])
def test_failed_coordinate_command_is_atomic(endpoint: JsonValue):
    env = FormEnv()
    form = Form(CoordinateCfg, {"sweep": Range.between(0, 10, 4)})
    raw: dict[str, JsonValue] = {"kind": "linrange", "stop": 10, "n": {"expr": "2 + 2"}}
    if endpoint is not None:
        raw["start"] = endpoint
    state = FormState(FormInput(form.model, {"sweep": raw}))
    target = state.describe(("sweep", "stop"), env).target
    commands = getattr(target, "commands", ())
    assert commands, "Coordinate commands must exist on endpoint fields"
    before = state.input.to_json()
    with pytest.raises(ValueError):
        state.run_command(("sweep", "stop"), commands[0].name, env)
    assert state.input.to_json() == before


def test_center_expression_restore_reset_and_single_point_contract():
    env = FormEnv()
    form = Form(CoordinateCfg, {"sweep": Range.between(3, 3, 1)})
    state = FormState(form.default_input(env))
    switch(state, env, "stop", "Switch to Center / Span")
    state.write({("sweep", "center"): {"expr": "1 + 2"}}, env)
    restored = FormState(FormInput.from_json(form.model, state.input.to_json()))
    assert restored.input.data == state.input.data
    result = restored.result(env)
    assert not isinstance(result, FormErrors)
    cfg = form.to_cfg(result, env)
    assert cfg == CoordinateCfg(sweep=SweepCfg(start=3, stop=3, expts=1, step=0))
    restored.write({("sweep", "span"): 2}, env)
    result = restored.result(env)
    assert not isinstance(result, FormErrors)
    assert isinstance(form.to_cfg(result, env), FormErrors)
    restored.reset(form.default_input(env))
    assert restored.input.data["sweep"] == {
        "kind": "linrange",
        "start": 3,
        "stop": 3,
        "n": 1,
    }


def test_failed_authored_endpoint_cannot_convert_using_inherited_default():
    env = FormEnv()
    form = Form(CoordinateCfg, {"sweep": Range.between(Expr("$missing.value"), 10, 4)})
    state = FormState(form.default_input(env))
    target = state.describe(("sweep", "stop"), env).target
    commands = getattr(target, "commands", ())
    assert commands, "Coordinate commands must exist"
    before = state.input.to_json()
    with pytest.raises(ValueError):
        state.run_command(("sweep", "stop"), commands[0].name, env)
    assert state.input.to_json() == before
