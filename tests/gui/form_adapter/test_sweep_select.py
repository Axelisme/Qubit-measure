"""Ordinary sweep selection preserves direct numeric intervals and sampling."""

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.form_adapter import Form, LogicalEdit, SweepInput
from zcu_tools.program.v2.sweep import SweepCfg


class SweepPlan(ConfigBase):
    sweep: SweepCfg
    other: float


class SweepGroup(ConfigBase):
    items: list[SweepPlan]


class HandwrittenSweep(FormModel):
    sweep: SweepInput
    other: float


@pytest.fixture(params=["generated", "indexed", "handwritten"])
def sweep_state(request):
    env = FormEnv()
    raw: dict[str, JsonValue] = {
        "sweep": {"kind": "linrange", "start": 10, "stop": 14, "n": 3},
        "other": {"expr": "1 / 0"},
    }
    if request.param == "indexed":
        model = Form(SweepGroup, {}).model
        data: dict[str, JsonValue] = {"items": [raw, raw.copy()]}
        path = ("items", 1, "sweep")
    else:
        model = (
            HandwrittenSweep
            if request.param == "handwritten"
            else Form(SweepPlan, {}).model
        )
        data = raw
        path = ("sweep",)
    return FormState(FormInput(model, data)), path, env


def replace_sweep(data, path, value):
    """Return detached expected input with only the mounted sweep replaced."""
    if len(path) == 1:
        data["sweep"] = value
    else:
        data["items"][1]["sweep"] = value
    return data


def test_select_center_preserves_interval_and_return(sweep_state):
    state, path, env = sweep_state
    before = state.input
    state.select(path, {"kind": "center"}, env)
    assert state.input.data == replace_sweep(
        before.data, path, {"kind": "center", "center": 12, "span": 4, "n": 3}
    )
    state.select(path, {"kind": "linrange"}, env)
    assert state.input.data == before.data
    assert state.base.same_as(before)


REPRESENTATIONS = [
    {"kind": "linrange", "start": 10, "stop": 14, "n": 3},
    {"kind": "center", "center": 12, "span": 4, "n": 3},
    {"kind": "linrange_step", "start": 10, "stop": 14, "step": 2},
    {"kind": "center_step", "center": 12, "span": 4, "step": 2},
]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (source, target)
        for source in REPRESENTATIONS
        for target in REPRESENTATIONS
        if source != target
    ],
    ids=lambda raw: raw["kind"],
)
def test_select_converts_both_dimensions_and_roundtrips(sweep_state, source, target):
    state, path, env = sweep_state
    state.write({path: source}, env)
    before, base = state.input, state.base
    events = []
    state.subscribe(lambda: events.append("changed"))
    state.select(path, {"kind": target["kind"]}, env)
    assert state.input.data == replace_sweep(before.data, path, target)
    assert events == ["changed"]
    state.select(path, {"kind": source["kind"]}, env)
    assert state.input.data == before.data
    assert state.base.same_as(base)
    assert events == ["changed", "changed"]


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        (
            {"kind": "linrange", "start": {"expr": "10"}, "stop": 14, "n": 3},
            "center",
            {"kind": "center", "n": 3},
        ),
        (
            {"kind": "center", "center": 12, "span": {"expr": "1 / 0"}, "n": 3},
            "linrange",
            {"kind": "linrange", "n": 3},
        ),
        (
            {"kind": "linrange", "stop": 14, "n": 3},
            "center",
            {"kind": "center", "n": 3},
        ),
        (
            {"kind": "linrange", "start": "bad", "stop": 14, "n": 3},
            "center",
            {"kind": "center", "n": 3},
        ),
        (
            {"kind": "linrange", "start": True, "stop": 14, "n": 3},
            "center",
            {"kind": "center", "n": 3},
        ),
        (
            {"kind": "linrange", "start": "10", "stop": 14, "n": 3},
            "center",
            {"kind": "center", "n": 3},
        ),
        (
            {"kind": "linrange", "start": 10, "stop": 14, "n": {"expr": "3"}},
            "center_step",
            {"kind": "center_step"},
        ),
        (
            {"kind": "linrange", "start": 10, "stop": 14, "n": {"expr": "1 / 0"}},
            "linrange_step",
            {"kind": "linrange_step", "start": 10, "stop": 14},
        ),
        (
            {"kind": "linrange", "start": 10, "stop": 14},
            "center_step",
            {"kind": "center_step"},
        ),
        (
            {"kind": "linrange", "start": 10, "stop": 14, "n": 1},
            "center_step",
            {"kind": "center_step"},
        ),
        (
            {"kind": "linrange", "start": 10, "stop": 14, "n": 0},
            "center_step",
            {"kind": "center_step"},
        ),
        (
            {"kind": "center_step", "center": 12, "span": 4, "step": 0},
            "linrange",
            {"kind": "linrange"},
        ),
        (
            {"kind": "linrange_step", "start": 10, "stop": 14, "step": -2},
            "center",
            {"kind": "center"},
        ),
        (
            {"kind": "linrange_step", "start": 10, "stop": 14, "step": 5},
            "center",
            {"kind": "center"},
        ),
        (
            {"kind": "center_step", "center": 12, "span": 4, "step": {"expr": "2"}},
            "center",
            {"kind": "center", "center": 12, "span": 4},
        ),
        (
            {
                "kind": "linrange",
                "start": {"expr": "10"},
                "stop": 14,
                "n": {"expr": "3"},
            },
            "center",
            {"kind": "center", "n": {"expr": "3"}},
        ),
    ],
)
def test_failed_select_falls_back_without_partial_conversion(
    sweep_state, source, target, expected
):
    state, path, env = sweep_state
    state.write({path: source}, env)
    before, base = state.input, state.base
    events = []
    state.subscribe(lambda: events.append("changed"))
    state.select(path, {"kind": target}, env)
    assert state.input.data == replace_sweep(before.data, path, expected)
    assert state.base.same_as(base)
    assert events == ["changed"]


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        (
            {"kind": "linrange_step", "start": 10, "stop": 14, "step": 1.5},
            "center",
            {"kind": "center", "center": 11.5, "span": 3, "n": 3},
        ),
        (
            {"kind": "center_step", "center": 12, "span": 4, "step": 1.5},
            "linrange",
            {"kind": "linrange", "start": 10, "stop": 13, "n": 3},
        ),
        (
            {"kind": "linrange", "start": 14, "stop": 10, "n": 3},
            "center_step",
            {"kind": "center_step", "center": 12, "span": -4, "step": -2},
        ),
        (
            {"kind": "linrange_step", "start": 1, "stop": 0, "step": -0.3},
            "center",
            {"kind": "center", "center": 0.55, "span": -0.9, "n": 4},
        ),
        (
            {"kind": "linrange_step", "start": 0, "stop": 0.3, "step": 0.1},
            "linrange",
            {"kind": "linrange", "start": 0, "stop": 0.3, "n": 4},
        ),
        (
            {"kind": "linrange", "start": 10, "stop": 10, "n": 1},
            "center_step",
            {"kind": "center_step", "center": 10, "span": 0, "step": 0},
        ),
        (
            {"kind": "center_step", "center": 10, "span": 0, "step": 0},
            "linrange",
            {"kind": "linrange", "start": 10, "stop": 10, "n": 1},
        ),
        (
            {"kind": "linrange", "start": 10, "stop": 14, "n": {"expr": "3"}},
            "center",
            {"kind": "center", "center": 12, "span": 4, "n": {"expr": "3"}},
        ),
    ],
)
def test_select_uses_command_sampling_rules_and_only_needed_values(
    sweep_state, source, target, expected
):
    state, path, env = sweep_state
    state.write({path: source}, env)
    state.select(path, {"kind": target}, env)
    data = state.input.data
    actual = data["sweep"] if len(path) == 1 else data["items"][1]["sweep"]
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name] == (
            pytest.approx(value) if isinstance(value, (int, float)) else value
        )


def test_select_unrepresentable_coordinates_fall_back_without_error(sweep_state):
    state, path, env = sweep_state
    source = {"kind": "linrange", "start": -1e308, "stop": 1e308, "n": 3}
    state.write({path: source}, env)
    before = state.input
    state.select(path, {"kind": "center"}, env)
    assert state.input.data == replace_sweep(
        before.data, path, {"kind": "center", "n": 3}
    )


def test_reselect_active_branch_keeps_invalid_raw(sweep_state):
    state, path, env = sweep_state
    source = {"kind": "center_step", "center": {"expr": "12"}, "span": "bad"}
    state.write({path: source}, env)
    before = state.input
    state.select(path, {"kind": "center_step"}, env)
    assert state.input.data == before.data


def test_complete_logical_sweep_edit_does_not_require_old_draft_to_convert():
    env = FormEnv()
    form = Form(SweepPlan, {})
    state = FormState(
        FormInput(form.model, {"sweep": {"kind": "center", "span": "bad"}})
    )
    before = state.input
    replacement: dict[str, JsonValue] = {
        "kind": "linrange",
        "start": {"expr": "10"},
        "stop": 14,
        "n": 3,
    }
    candidate = form.prepare_logical_edits(
        state.input, (LogicalEdit(("sweep",), replacement),), env
    )
    assert state.input.same_as(before)
    state.install_input(candidate.input)
    assert state.input.data == {"sweep": replacement}
