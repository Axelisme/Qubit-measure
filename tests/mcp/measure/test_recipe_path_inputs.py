"""Declared object-list inputs preserve typed paths through public recipe tools."""

from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import TypedDict

import pytest
from jsonschema import Draft7Validator
from zcu_tools.mcp.measure.recipe import (
    RecipeGenerator,
    RecipeInputSchema,
    RecipeInputValue,
    RecipePropertySchema,
    RecipeSession,
)

from ._recipe_execution_support import definition
from ._support import MeasureClient, make_client


class PathOverride(TypedDict):
    path: list[str | int]
    value: RecipeInputValue


class PathInputs(TypedDict):
    participants: list[str]
    overrides: list[PathOverride]
    memo: str | None


def path_schema(value_schema: RecipePropertySchema) -> RecipeInputSchema:
    """Declare paths as data, without interpreting fields or resolving indices."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["participants", "overrides"],
        "properties": {
            "participants": {"type": "array", "items": {"type": "string"}},
            "overrides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "value"],
                    "properties": {
                        "path": {
                            "type": "array",
                            "items": {"type": ["string", "integer"]},
                            "minItems": 1,
                        },
                        "value": value_schema,
                    },
                },
            },
            "memo": {"type": ["string", "null"]},
        },
    }


def path_client(
    tmp_path: Path,
    received: list[PathInputs],
    value_schema: RecipePropertySchema,
) -> MeasureClient:
    """Register an author that records inputs without using a GUI or lab recipe."""

    def author(
        session: RecipeSession,
        *,
        participants: list[str],
        overrides: list[PathOverride],
        memo: str | None = "author-default",
    ) -> RecipeGenerator:
        received.append(
            {"participants": participants, "overrides": overrides, "memo": memo}
        )
        yield from ()

    try:
        return make_client(
            tmp_path, recipes=(definition(author, path_schema(value_schema)),)
        )
    except ValueError as error:
        pytest.fail(
            f"Public recipe registration rejected declared object list: {error}"
        )


def test_recipe_tool_delivers_detached_typed_paths_and_number_values(tmp_path):
    received: list[PathInputs] = []
    gain_path: list[str | int] = ["tones", 0, "drive", "gain"]
    count_path: list[str | int] = ["tones", 2, "sweep", "points"]
    participants = ["Q3", "Q1"]
    overrides = [
        {"path": gain_path, "value": 1},
        {"path": count_path, "value": 3.5},
        {"path": ["tones", "1", "drive", "gain"], "value": 2},
    ]
    arguments = {"participants": participants, "overrides": overrides}
    original = deepcopy(arguments)
    client = path_client(tmp_path, received, {"type": "number"})
    with closing(client.context.session):
        # Exercise the exposed tool schema on the same request sent to admission.
        Draft7Validator(client.tools["sample"]["inputSchema"]).validate(arguments)
        reply = client.call("sample", arguments)
        assert not reply.is_error and reply.data["status"] == "finished"
    assert received == [{**original, "memo": "author-default"}]
    delivered = received[0]["overrides"]
    assert [type(segment) for segment in delivered[0]["path"]] == [str, int, str, str]
    assert type(delivered[0]["value"]) is float
    assert type(delivered[1]["value"]) is float
    assert type(delivered[2]["path"][1]) is str
    delivered[0]["path"][1] = 99
    delivered[0]["value"] = 9.0
    delivered.append({"path": ["later"], "value": None})
    received[0]["participants"].append("later")
    assert arguments == original
    participants.clear()
    gain_path.clear()
    overrides.clear()
    assert received[0]["overrides"][1] == {
        "path": ["tones", 2, "sweep", "points"],
        "value": 3.5,
    }


@pytest.mark.parametrize(
    ("value_schema", "value"),
    [
        pytest.param({"type": "integer"}, 3, id="integer"),
        pytest.param({"type": "boolean"}, False, id="boolean"),
        pytest.param({"type": "string"}, "0.3", id="string"),
        pytest.param({"type": ["number", "null"]}, None, id="null"),
    ],
)
def test_recipe_tool_preserves_authored_value_types_and_explicit_null(
    tmp_path, value_schema, value
):
    received: list[PathInputs] = []
    client = path_client(tmp_path, received, value_schema)
    with closing(client.context.session):
        reply = client.call(
            "sample",
            {
                "participants": ["Q3", "Q1"],
                "overrides": [
                    {"path": ["tones", 1, "sweep", "points"], "value": value}
                ],
                "memo": None,
            },
        )
        assert not reply.is_error and reply.data["status"] == "finished"
    assert received[0]["overrides"][0]["value"] == value
    assert type(received[0]["overrides"][0]["value"]) is type(value)
    assert received[0]["memo"] is None


def test_recipe_tool_rejects_float_for_declared_integer_override_value(tmp_path):
    received: list[PathInputs] = []
    client = path_client(tmp_path, received, {"type": "integer"})
    with closing(client.context.session):
        reply = client.call(
            "sample",
            {
                "participants": ["Q1"],
                "overrides": [{"path": ["tones", 0, "sweep", "points"], "value": 3.0}],
            },
        )
        assert reply.is_error and reply.data["status"] == "failed"
        assert "value" in reply.data["error"]["message"]
        assert received == []
        assert client.transport.sent == []


def test_recipe_tool_detaches_declared_nested_values_inside_override_items(tmp_path):
    received: list[PathInputs] = []
    value_schema: RecipePropertySchema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "drive": {
                "type": "object",
                "properties": {"gain": {"type": "number"}},
                "additionalProperties": False,
            },
            "sweep": {"type": "array", "items": {"type": "integer"}},
        },
    }
    sweep = [1, 2]
    arguments = {
        "participants": ["Q1"],
        "overrides": [
            {"path": ["tones", 0], "value": {"drive": {"gain": 1}, "sweep": sweep}}
        ],
    }
    original = deepcopy(arguments)
    client = path_client(tmp_path, received, value_schema)
    with closing(client.context.session):
        reply = client.call("sample", arguments)
        assert not reply.is_error and reply.data["status"] == "finished"
    sweep.append(3)
    assert received == [{**original, "memo": "author-default"}]


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        pytest.param({}, "overrides", id="object-not-list"),
        pytest.param([1], "overrides.0", id="scalar-item"),
        pytest.param([[]], "overrides.0", id="array-item"),
        pytest.param([{"value": 1}], "path", id="missing-path"),
        pytest.param([{"path": ["tones", 0]}], "value", id="missing-value"),
        pytest.param([{"path": "tones.0", "value": 1}], "path", id="string-path"),
        pytest.param([{"path": [], "value": 1}], "path", id="empty-path"),
        pytest.param([{"path": [True], "value": 1}], "path", id="bool-segment"),
        pytest.param([{"path": [0.0], "value": 1}], "path", id="float-segment"),
        pytest.param([{"path": [None], "value": 1}], "path", id="null-segment"),
        pytest.param([{"path": [{}], "value": 1}], "path", id="object-segment"),
        pytest.param([{"path": [[0]], "value": 1}], "path", id="array-segment"),
        pytest.param([{"path": ["tones"], "value": "1"}], "value", id="string-number"),
        pytest.param([{"path": ["tones"], "value": True}], "value", id="bool-number"),
        pytest.param([{"path": ["tones"], "value": None}], "value", id="nonnullable"),
        pytest.param([{"path": ["tones"], "value": {}}], "value", id="object-value"),
        pytest.param(
            [{"path": ["tones"], "value": float("inf")}], "value", id="infinite"
        ),
        pytest.param([{"path": ["tones"], "value": float("nan")}], "value", id="nan"),
        pytest.param([{"path": ["tones"], "value": 10**1000}], "value", id="overflow"),
        pytest.param(
            [{"path": ["tones"], "value": 1, "extra": 2}], "extra", id="unknown-field"
        ),
    ],
)
def test_recipe_tool_rejects_invalid_override_items_before_callback(
    tmp_path, overrides, path
):
    received: list[PathInputs] = []
    client = path_client(tmp_path, received, {"type": "number"})
    with closing(client.context.session):
        reply = client.call("sample", {"participants": ["Q1"], "overrides": overrides})
        assert reply.is_error and reply.data["status"] == "failed"
        assert path in reply.data["error"]["message"]
        full = client.call(
            "status", {"execution": reply.data["execution"], "detail": "full"}
        )
        assert full["error"]["phase"] == "preparing"
        assert received == []
        assert client.transport.sent == []
