"""Sampling commands, validated effective coordinates and authored draft isolation."""

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import ConfigBase, Range
from zcu_tools.gui.form import (
    FormCommandError,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
)
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2.sweep import SweepCfg


class SamplingCfg(ConfigBase):
    sweep: SweepCfg
    other: float = 1


def convert(state: FormState[FormModel], field: str, label: str) -> None:
    env = FormEnv()
    commands = getattr(state.describe(("sweep", field), env).target, "commands", ())
    command = next((item for item in commands if item.label == label), None)
    assert command is not None, f"Missing sampling command {label}"
    state.run_command(("sweep", field), command.name, env)


def step_state(start: float = 0, stop: float = 1, n: int = 5):
    form = Form(SamplingCfg, {"sweep": Range.between(start, stop, n)})
    state = FormState(form.default_input(FormEnv()))
    convert(state, "n", "Switch to Step")
    return form, state


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize(
    ("start", "stop", "step", "n", "actual_stop"),
    [
        (0, 1, 0.3, 4, 0.9),
        (0, 1, 0.26, 4, 0.78),
        (0, 0.3, 0.1, 4, 0.3),
        (0, 1, 0.1, 11, 1),
        (1, 0, -0.3, 4, 0.1),
    ],
)
def test_step_projection_preserves_authored_input(
    center, start, stop, step, n, actual_stop
):
    env = FormEnv()
    form, state = step_state(start, stop)
    if center:
        convert(state, "start", "Switch to Center / Span")
    state.write({("sweep", "step"): {"expr": str(step)}}, env)
    authored = state.input.to_json()
    preview = state.preview(env)
    assert preview.errors is None
    coord = "center" if center else "stop"
    assert preview.values[("sweep", coord)] == pytest.approx(
        (start + actual_stop) / 2 if center else actual_stop
    )
    if center:
        assert preview.values[("sweep", "span")] == pytest.approx(actual_stop - start)
    result = state.result(env)
    assert not isinstance(result, FormErrors)
    cfg = form.to_cfg(result, env)
    assert not isinstance(cfg, FormErrors)
    assert cfg.sweep.start == start
    assert cfg.sweep.stop == pytest.approx(actual_stop)
    assert cfg.sweep.step == step
    assert cfg.sweep.expts == n
    assert state.input.to_json() == authored
    convert(state, "step", "Switch to Expts")
    assert state.input.data["sweep"] == {
        "kind": "center" if center else "linrange",
        **(
            {
                "center": pytest.approx((start + actual_stop) / 2),
                "span": pytest.approx(actual_stop - start),
            }
            if center
            else {"start": start, "stop": pytest.approx(actual_stop)}
        ),
        "n": n,
    }


@pytest.mark.parametrize("step", [0, -0.1, 2, {"expr": "1 / 0"}])
def test_invalid_step_draft_rejects_result_and_atomic_conversion(step: JsonValue):
    _, state = step_state()
    state.write({("sweep", "step"): step}, FormEnv())
    authored = state.input.to_json()
    assert state.preview(FormEnv()).errors is not None
    assert isinstance(state.result(FormEnv()), FormErrors)
    with pytest.raises(FormCommandError):
        convert(state, "step", "Switch to Expts")
    assert state.input.to_json() == authored


@pytest.mark.parametrize(("start", "stop", "n"), [(2, 2, 3), (0, 1, 1)])
def test_inconsistent_expts_cannot_be_repaired_by_sampling_switch(start, stop, n):
    form = Form(SamplingCfg, {"sweep": Range.between(0, 1, 3)})
    state = FormState(form.default_input(FormEnv()))
    state.write(
        {("sweep", "start"): start, ("sweep", "stop"): stop, ("sweep", "n"): n},
        FormEnv(),
    )
    commands = getattr(state.describe(("sweep", "n"), FormEnv()).target, "commands", ())
    assert commands, "Expts must expose sampling conversion"
    before = state.input.to_json()
    with pytest.raises(FormCommandError):
        state.run_command(("sweep", "n"), commands[0].name, FormEnv())
    assert state.input.to_json() == before


def test_sampling_switch_evaluates_participants_only_and_roundtrips_single_point():
    _, state = step_state(2, 2, 1)
    convert(state, "start", "Switch to Center / Span")
    assert state.input.data["sweep"] == {
        "kind": "center_step",
        "center": 2,
        "span": 0,
        "step": 0,
    }
    state.write(
        {
            ("sweep", "center"): {"expr": "1 + 1"},
            ("sweep", "step"): {"expr": "0 * 2"},
            ("other",): {"expr": "1 / 0"},
        },
        FormEnv(),
    )
    convert(state, "step", "Switch to Expts")
    assert state.input.data == {
        "sweep": {"kind": "center", "center": 2, "span": 0, "n": 1},
        "other": {"expr": "1 / 0"},
    }


@pytest.mark.parametrize("field", ["start", "stop", "n"])
def test_sampling_failure_keeps_all_expression_sources(field):
    form = Form(SamplingCfg, {"sweep": Range.between(0, 1, 3)})
    state = FormState(form.default_input(FormEnv()))
    state.write(
        {
            ("sweep", "start"): {"expr": "0 * 2"},
            ("sweep", "stop"): {"expr": "1 + 0"},
            ("sweep", "n"): {"expr": "2 + 1"},
            ("sweep", field): {"expr": "1 / 0"},
        },
        FormEnv(),
    )
    commands = getattr(state.describe(("sweep", "n"), FormEnv()).target, "commands", ())
    assert commands, "Expts must expose sampling conversion"
    before = state.input.to_json()
    with pytest.raises(FormCommandError):
        state.run_command(("sweep", "n"), commands[0].name, FormEnv())
    assert state.input.to_json() == before


def test_valid_step_preview_still_projects_when_unrelated_field_is_invalid():
    _, state = step_state()
    state.write({("sweep", "step"): 0.3, ("other",): {"expr": "1 / 0"}}, FormEnv())
    preview = state.preview(FormEnv())
    assert preview.errors is not None
    assert preview.values[("sweep", "stop")] == pytest.approx(0.9)


def test_sampling_success_discards_all_participating_expressions():
    form = Form(SamplingCfg, {"sweep": Range.between(0, 1, 3)})
    state = FormState(form.default_input(FormEnv()))
    state.write(
        {
            ("sweep", "start"): {"expr": "0 * 2"},
            ("sweep", "stop"): {"expr": "1 + 0"},
            ("sweep", "n"): {"expr": "2 + 1"},
        },
        FormEnv(),
    )
    convert(state, "n", "Switch to Step")
    assert state.input.data["sweep"] == {
        "kind": "linrange_step",
        "start": 0,
        "stop": 1,
        "step": 0.5,
    }


def test_missing_step_draft_is_not_completed_by_conversion():
    form, state = step_state()
    raw = state.input.data
    sweep = raw["sweep"]
    assert isinstance(sweep, dict)
    del sweep["step"]
    state = FormState(FormInput(form.model, raw))
    before = state.input.to_json()
    with pytest.raises(FormCommandError):
        convert(state, "step", "Switch to Expts")
    assert state.input.to_json() == before


def test_step_coordinate_switch_ignores_sampling_and_restore_preserves_new_expression():
    form, state = step_state()
    state.write({("sweep", "step"): {"expr": "1 / 0"}}, FormEnv())
    convert(state, "start", "Switch to Center / Span")
    state.write({("sweep", "center"): {"expr": "0.25 + 0.25"}}, FormEnv())
    restored = FormState(FormInput.from_json(form.model, state.input.to_json()))
    assert restored.input.data == state.input.data
    assert restored.input.data["sweep"] == {
        "kind": "center_step",
        "center": {"expr": "0.25 + 0.25"},
        "span": 1,
        "step": {"expr": "1 / 0"},
    }
    convert(restored, "span", "Switch to Range")
    assert restored.input.data["sweep"] == {
        "kind": "linrange_step",
        "start": 0,
        "stop": 1,
        "step": {"expr": "1 / 0"},
    }
    restored.reset(form.default_input(FormEnv()))
    assert restored.input.data["sweep"] == {
        "kind": "linrange",
        "start": 0,
        "stop": 1,
        "n": 5,
    }
