"""Persistence and isolation through the public form input envelope."""

import pytest
from pydantic import JsonValue
from zcu_tools.gui.form import FormInput, FormModel


class InputForm(FormModel):
    """Ordinary values used to observe incomplete saved input."""

    required: float
    optional: float | None = None
    samples: tuple[float, ...] = ()


def test_input_round_trip_preserves_missing_null_and_scalar_envelopes() -> None:
    data: dict[str, JsonValue] = {
        "optional": None,
        "samples": [1.0, {"expr": "seed * 2"}],
        "amplitude": {"__complex__": [1.0, -2.0]},
    }
    original = FormInput(InputForm, data)
    data["samples"] = []
    exposed = original.data
    exposed["optional"] = 42.0

    restored = FormInput.from_json(InputForm, original.to_json())
    assert restored.model is InputForm
    assert restored.data == {
        "optional": None,
        "samples": [1.0, {"expr": "seed * 2"}],
        "amplitude": {"__complex__": [1.0, -2.0]},
    }
    assert "required" not in restored.data
    assert original.data == restored.data


@pytest.mark.parametrize("text", ["[]", "null", "{", '{"required": NaN}'])
def test_saved_input_rejects_invalid_json_envelopes(text: str) -> None:
    with pytest.raises(ValueError, match="JSON|json|object|finite"):
        FormInput.from_json(InputForm, text)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_input_rejects_nonfinite_json_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="finite|JSON|json"):
        FormInput(InputForm, {"required": value})


def test_input_keeps_unknown_fields_and_invalid_values_for_restoration() -> None:
    data: dict[str, JsonValue] = {
        "required": {"expr": 42},
        "optional": "not a number",
        "old_name": "saved before a model change",
    }
    assert (
        FormInput.from_json(InputForm, FormInput(InputForm, data).to_json()).data
        == data
    )
