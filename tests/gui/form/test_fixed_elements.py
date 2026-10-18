"""Fixed captured identities, typed editing and independent participation."""

import pytest
from pydantic import JsonValue
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPathError,
    FormState,
)

from tests.gui.form._fixed_models import named_form, named_input


def expect_errors(result: FormModel | FormErrors) -> FormErrors:
    """Require the public error outcome for a model-invalid draft."""
    assert isinstance(result, FormErrors)
    return result


def test_describe_and_edit_original_element_positions():
    state = FormState(named_input(named_form()))
    env = FormEnv()
    description = state.describe(("tones",), env)
    assert description.target is not None
    assert description.target.elements is not None
    assert [(item.index, item.name) for item in description.target.elements] == [
        (0, "A"),
        (1, "B"),
    ]
    assert [item.path for item in state.describe(("tones", 1), env).children] == [
        ("tones", 1, "label"),
        ("tones", 1, "gain"),
    ]
    identity = state.describe(("tones", 1, "label"), env).target
    assert identity is not None
    assert identity.fixed
    state.write({("tones", 1, "gain"): {"expr": "3 * 2"}}, env)
    assert state.preview(env).values[("tones", 1, "gain")] == 6
    assert state.input.data["participants"] == ["B"]
    assert state.preview(env).values[("tones", 0, "gain")] == 1
    for path in [
        ("tones", "1", "gain"),
        ("tones", -1),
        ("tones", 2),
        ("tones", True),
        ("tones", 0, "unknown"),
    ]:
        with pytest.raises(FormPathError) as caught:
            state.describe(path, env)
        assert caught.value.path == path
        before = state.input
        with pytest.raises(ValueError, match="path"):
            state.write({path: 9}, env)
        assert state.input.same_as(before)


def test_fixed_identity_admission_is_atomic_for_container_and_prepared_input():
    model = named_form()
    state = FormState(named_input(model))
    original = state.input
    notifications = []
    state.subscribe(lambda: notifications.append(state.input))
    replacements: list[JsonValue] = [
        [],
        [{"label": "A", "gain": 1}],
        [{"label": "B", "gain": 2}, {"label": "A", "gain": 1}],
        [{"label": "C", "gain": 1}, {"label": "B", "gain": 2}],
    ]
    for replacement in replacements:
        with pytest.raises(ValueError, match="identity"):
            state.write({("tones",): replacement}, FormEnv())
        with pytest.raises(ValueError, match="identity"):
            state.install_input(
                FormInput(model, {"tones": replacement, "participants": ["B"]})
            )
        assert state.input.same_as(original)
    with pytest.raises(ValueError, match="identity"):
        state.write({("tones", 0, "label"): "C", ("tones", 1, "gain"): 8}, FormEnv())
    assert notifications == []
    state.write(
        {("tones",): [{"label": "A", "gain": 4}, {"label": "B", "gain": -2}]}, FormEnv()
    )
    assert len(notifications) == 1
    assert ("tones", 1, "gain") in expect_errors(state.result(FormEnv())).messages


def test_duplicate_element_names_rejected_with_original_indices():
    model = named_form()
    with pytest.raises(ValueError, match=r"A.*0.*1"):
        FormState(
            FormInput(
                model,
                {
                    "tones": [{"label": "A", "gain": 1}, {"label": "A", "gain": 2}],
                    "participants": ["A"],
                },
            )
        )


def test_participants_validate_without_filtering_unselected_elements():
    state = FormState(named_input(named_form()))
    env = FormEnv()
    participant_inputs: list[JsonValue] = [[], ["A", "A"], ["missing"]]
    for participants in participant_inputs:
        state.write({("participants",): participants}, env)
        result = state.result(env)
        assert isinstance(result, FormErrors)
        assert any(path[:1] == ("participants",) for path in result.messages)
    state.write({("participants",): ["B", "A"]}, env)
    valid = state.result(env)
    assert isinstance(valid, FormModel)
    assert valid.model_dump()["participants"] == ["B", "A"]
    state.write({("participants",): ["B"], ("tones", 0, "gain"): -1}, env)
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert ("tones", 0, "gain") in result.messages
    state.write({("tones", 0, "gain"): {"expr": "unfinished +"}}, env)
    assert ("tones", 0, "gain") in expect_errors(state.result(env)).messages
