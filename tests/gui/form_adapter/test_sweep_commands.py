"""Sweep row commands are shared by headless callers and the Qt view."""

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import ConfigBase, Range
from zcu_tools.gui.form import FormCommandError, FormEnv, FormInput, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2.sweep import SweepCfg


class SweepPlan(ConfigBase):
    sweep: SweepCfg
    other: float = 1


@pytest.mark.parametrize("sampling", [None, {"expr": "1 / 0"}, {"expr": "2 + 2"}])
def test_sweep_row_command_preserves_missing_or_invalid_sampling(sampling):
    env = FormEnv()
    form = Form(SweepPlan, {"sweep": Range(10, 4, 4)})
    raw: dict[str, JsonValue] = {"kind": "linrange", "start": 10, "stop": 4}
    if sampling is not None:
        raw["n"] = sampling
    state = FormState(FormInput(form.model, {"sweep": raw, "other": {"expr": "1 / 0"}}))
    target = state.describe(("sweep",), env).target
    assert target is not None
    assert {command.name for command in target.commands} == {"to_center", "to_step"}, (
        "Sweep actions belong to the sweep row"
    )
    state.run_command(("sweep",), "to_center", env)
    expected: dict[str, JsonValue] = {"kind": "center", "center": 7, "span": -6}
    if sampling is not None:
        expected["n"] = sampling
    assert state.input.data == {"sweep": expected, "other": {"expr": "1 / 0"}}
    child = state.describe(("sweep", "center"), env).target
    assert child is not None
    assert child.commands == ()


def test_sweep_leaf_cannot_replace_parent():
    env = FormEnv()
    form = Form(SweepPlan, {"sweep": Range(10, 4, 4)})
    state = FormState(form.default_input(env))
    before = state.input.to_json()
    with pytest.raises(FormCommandError):
        state.run_command(("sweep", "start"), "to_center", env)
    assert state.input.to_json() == before
