"""Public recipe input boundary, independent of GUI side effects."""

from copy import deepcopy

import pytest
from zcu_tools.mcp.measure.recipe import RecipeInputSchema
from zcu_tools.mcp.measure.recipe_inputs import RecipeInputs


def input_schema() -> RecipeInputSchema:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["label"],
        "properties": {
            "label": {"type": "string", "minLength": 1},
            "frequency": {"type": ["number", "null"]},
            "points": {"type": ["integer", "null"]},
            "enabled": {"type": "boolean"},
            "values": {
                "type": ["array", "null"],
                "items": {"type": "number"},
                "minItems": 2,
            },
        },
    }


def test_recipe_inputs_preserve_null_defaults_and_normalize_numeric_arrays():
    schema = input_schema()
    inputs = RecipeInputs(schema)
    arguments = {
        "label": "capture",
        "frequency": 10,
        "points": 3,
        "enabled": True,
        "values": [1, 2.5],
    }
    original = deepcopy(arguments)

    normalized = inputs.normalize(arguments)

    assert normalized == arguments
    assert isinstance(normalized["frequency"], float)
    assert isinstance(normalized["points"], int)
    assert isinstance(normalized["values"], list)
    assert all(isinstance(value, float) for value in normalized["values"])
    assert arguments == original
    assert inputs.normalize({"label": "capture"}) == {"label": "capture"}
    assert inputs.normalize(
        {"label": "capture", "frequency": None, "values": None}
    ) == {"label": "capture", "frequency": None, "values": None}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "label"),
        ({"label": ""}, "label"),
        ({"label": "capture", "extra": 1}, "extra"),
        ({"label": "capture", "points": 3.0}, "points"),
        ({"label": "capture", "points": True}, "points"),
        ({"label": "capture", "frequency": True}, "frequency"),
        ({"label": "capture", "enabled": 1}, "enabled"),
        ({"label": "capture", "values": [1]}, "values"),
        ({"label": "capture", "values": [1, "2"]}, "values"),
        ({"label": "capture", "values": [1, True]}, "values"),
        ({"label": "capture", "frequency": float("nan")}, "frequency"),
        ({"label": "capture", "frequency": float("inf")}, "frequency"),
        ({"label": "capture", "values": [1, float("-inf")]}, "values"),
        ({"label": "capture", "frequency": 10**1000}, "frequency"),
    ],
)
def test_recipe_inputs_reject_invalid_values_with_the_parameter_name(
    arguments, message
):
    inputs = RecipeInputs(input_schema())

    with pytest.raises(ValueError, match=message):
        inputs.normalize(arguments)


def test_recipe_inputs_do_not_alias_arguments_or_the_authored_schema():
    schema = input_schema()
    inputs = RecipeInputs(schema)
    schema["properties"]["values"]["minItems"] = 0
    schema["properties"]["label"]["minLength"] = 0
    arguments = {"label": "capture", "values": [1, 2]}
    normalized = inputs.normalize(arguments)
    assert isinstance(normalized["values"], list)
    normalized["values"].append(3)

    assert arguments["values"] == [1, 2]
    with pytest.raises(ValueError, match="values"):
        inputs.normalize({"label": "capture", "values": []})
    with pytest.raises(ValueError, match="label"):
        inputs.normalize({"label": ""})


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "not_a_json_type"}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "string", "minLength": -1}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "array", "minItems": -1}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "object"}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "array", "items": {"type": "array"}}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": ["not_declared"],
        },
    ],
)
def test_recipe_inputs_reject_invalid_declarations_at_construction(schema):
    with pytest.raises(ValueError, match="schema"):
        RecipeInputs(schema)


def test_recipe_inputs_validate_even_explicitly_allowed_extra_parameters():
    schema: RecipeInputSchema = {
        "type": "object",
        "additionalProperties": True,
        "properties": {},
    }
    inputs = RecipeInputs(schema)
    assert inputs.normalize({"extra": [None, True, "value", 3]}) == {
        "extra": [None, True, "value", 3]
    }
    with pytest.raises(ValueError, match="extra"):
        inputs.normalize({"extra": {"nested": 1}})
    with pytest.raises(ValueError, match="extra"):
        inputs.normalize({"extra": [[1]]})
