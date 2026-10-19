"""Opt-in logarithmic integer lists through the generated Form boundary."""

import pytest
from pydantic import Field, JsonValue
from zcu_tools import cfg_model
from zcu_tools.gui.form import FormConversionError, FormEnv, FormErrors, FormState
from zcu_tools.gui.form_adapter import Form


class IntegerPlan(cfg_model.ConfigBase):
    depths: list[int]
    other: float = 1


def make_state(maximum: int = 128, points: int = 9):
    env = FormEnv()
    form = Form(
        IntegerPlan, {"depths": cfg_model.LogPoints(max=maximum, points=points)}
    )
    return form, FormState(form.default_input(env)), env


def cfg_result(form, state, env):
    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, env)
    assert not isinstance(cfg, FormErrors), cfg
    return cfg


@pytest.mark.parametrize(
    ("maximum", "points", "expected"),
    [
        (128, 9, [0, 1, 2, 4, 8, 16, 32, 64, 128]),
        (3, 9, [0, 1, 2, 3]),
        (0, 9, [0]),
        (1, 9, [0, 1]),
        (9, 5, [0, 1, 2, 4, 9]),
    ],
)
def test_log_points_lower_to_rounded_unique_integer_list(maximum, points, expected):
    form, state, env = make_state(maximum, points)
    authored = state.input.to_json()
    assert cfg_result(form, state, env).depths == expected
    assert state.preview(env).values[("depths", "points")] == len(expected)
    assert state.input.to_json() == authored


@pytest.mark.parametrize(("maximum", "expected"), [(128, [0, 128]), (1, [0, 1]), (0, [0])])
def test_two_points_keep_zero_and_the_configured_maximum(maximum, expected):
    form, state, env = make_state(maximum, 2)
    assert cfg_result(form, state, env).depths == expected


@pytest.mark.parametrize("maximum", [0, 128])
def test_fewer_than_two_requested_points_are_invalid_even_for_zero_maximum(maximum):
    _, state, env = make_state(maximum, 1)
    assert isinstance(state.result(env), FormErrors)


def test_single_value_list_to_log_installs_invalid_draft_without_inventing_points():
    _, state, env = make_state()
    state.write({("depths",): {"kind": "list", "values": [5]}}, env)
    state.select(("depths",), {"kind": "log"}, env)
    assert state.input.data["depths"] == {"kind": "log", "max": 5, "points": 1}
    assert isinstance(state.result(env), FormErrors)


def test_log_list_conversion_evaluates_only_participants_and_uses_actual_count():
    form, state, env = make_state(3, 9)
    state.write({("other",): {"expr": "1 / 0"}}, env)
    state.select(("depths",), {"kind": "list"}, env)
    assert state.input.data["depths"] == {"kind": "list", "values": [0, 1, 2, 3]}
    state.write({("depths", "values"): [0, 2, 7, 16]}, env)
    state.select(("depths",), {"kind": "log"}, env)
    assert state.input.data["depths"] == {"kind": "log", "max": 16, "points": 4}
    state.write({("other",): 1}, env)
    assert cfg_result(form, state, env).depths == [0, 1, 4, 16]


@pytest.mark.parametrize("value", [-1, {"expr": "1 / 0"}])
def test_failed_log_conversion_preserves_draft(value: JsonValue):
    _, state, env = make_state()
    state.write({("depths", "max"): value}, env)
    before = state.input.to_json()
    with pytest.raises(FormConversionError):
        state.select(("depths",), {"kind": "list"}, env)
    assert state.input.to_json() == before


@pytest.mark.parametrize("values", [[], [-1, 2], [0, 1.5], [True, 2]])
def test_invalid_explicit_lists_remain_invalid(values: JsonValue):
    _, state, env = make_state()
    state.write({("depths",): {"kind": "list", "values": values}}, env)
    assert isinstance(state.result(env), FormErrors)


class ConstrainedIntegerPlan(cfg_model.ConfigBase):
    depths: list[int] = Field(min_length=1)


def test_native_list_constraints_apply_after_lowering_not_to_input_model():
    env = FormEnv()
    form = Form(
        ConstrainedIntegerPlan, {"depths": cfg_model.LogPoints(max=3, points=9)}
    )
    state = FormState(form.default_input(env))
    assert cfg_result(form, state, env).depths == [0, 1, 2, 3]


def test_plain_integer_list_remains_a_whole_leaf():
    form = Form(IntegerPlan, {"depths": [0, 3, 7]})
    env = FormEnv()
    state = FormState(form.default_input(env))
    target = state.describe(("depths",), env).target
    assert target is not None
    assert target.options is None
    assert cfg_result(form, state, env).depths == [0, 3, 7]
