"""Declared nested recipe inputs reach synthetic authors through public tools."""

from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import TypedDict

import pytest
from zcu_tools.mcp.measure.recipe import (
    RecipeGenerator,
    RecipeInputSchema,
    RecipeSession,
)

from ._recipe_execution_support import definition
from ._support import MeasureClient, make_client


class DriveOverrides(TypedDict, total=False):
    gain: float
    enabled: bool
    label: str


class SweepOverrides(TypedDict, total=False):
    start: float
    stop: float
    points: int
    values: list[float]


class ToneOverrides(TypedDict, total=False):
    drive: DriveOverrides | None
    sweep: SweepOverrides


class AuthorInputs(TypedDict):
    participants: list[str]
    overrides: dict[str, ToneOverrides]
    rounds: int


def nested_schema() -> RecipeInputSchema:
    """Declare name-keyed overrides without any lab or Form dependency."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["participants", "overrides"],
        "properties": {
            "participants": {"type": "array", "items": {"type": "string"}},
            "overrides": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "drive": {
                            "type": ["object", "null"],
                            "additionalProperties": False,
                            "properties": {
                                "gain": {"type": "number"},
                                "enabled": {"type": "boolean"},
                                "label": {"type": "string"},
                            },
                        },
                        "sweep": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["points"],
                            "properties": {
                                "start": {"type": "number"},
                                "stop": {"type": "number"},
                                "points": {"type": "integer"},
                                "values": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                },
                            },
                        },
                    },
                },
            },
            "rounds": {"type": "integer"},
        },
    }


def nested_client(tmp_path: Path, received: list[AuthorInputs]) -> MeasureClient:
    """Bind a recording author; rejection of its valid declaration is a defect."""

    def author(
        session: RecipeSession,
        *,
        participants: list[str],
        overrides: dict[str, ToneOverrides],
        rounds: int = 7,
    ) -> RecipeGenerator:
        received.append(
            {"participants": participants, "overrides": overrides, "rounds": rounds}
        )
        yield from ()

    try:
        return make_client(tmp_path, recipes=(definition(author, nested_schema()),))
    except ValueError as error:
        pytest.fail(
            f"Public recipe registration rejected declared nested inputs: {error}"
        )


def test_nested_recipe_tool_delivers_detached_typed_overrides_to_author(tmp_path):
    received: list[AuthorInputs] = []
    arguments = {
        "participants": ["Q3", "Q1"],
        "overrides": {
            "Q1": {
                "drive": {"gain": 1, "enabled": True, "label": "first"},
                "sweep": {
                    "start": 100,
                    "stop": 102.5,
                    "points": 3,
                    "values": [100, 102],
                },
            },
            "Q2": {"drive": None},
            "Q3": {},
        },
        "rounds": 7,
    }
    original = deepcopy(arguments)
    client = nested_client(tmp_path, received)
    with closing(client.context.session):
        reply = client.call("sample", arguments)
        assert not reply.is_error and reply.data["status"] == "finished"
    assert received == [original]
    # JSON spelling alone cannot distinguish 1 from 1.0; the author must get floats.
    q1 = received[0]["overrides"]["Q1"]
    drive = q1.get("drive")
    assert drive is not None and type(drive.get("gain")) is float
    sweep = q1.get("sweep")
    assert sweep is not None and type(sweep.get("start")) is float
    assert type(sweep.get("points")) is int
    values = sweep.get("values")
    assert values is not None and all(type(value) is float for value in values)
    received[0]["participants"].append("later")
    values.append(103.0)
    assert arguments == original


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        pytest.param("not-an-object", "overrides", id="scalar-overrides"),
        pytest.param({"Q1": []}, "overrides.Q1", id="list-element"),
        pytest.param({"Q1": {"drive": []}}, "overrides.Q1.drive", id="list-drive"),
        pytest.param({"Q1": {"drive": {"unknown": 1}}}, "unknown", id="unknown-leaf"),
        pytest.param({"Q1": {"drive": {"gain": "1"}}}, "gain", id="string-number"),
        pytest.param({"Q1": {"drive": {"gain": True}}}, "gain", id="bool-number"),
        pytest.param({"Q1": {"drive": {"gain": float("inf")}}}, "gain", id="nonfinite"),
        pytest.param({"Q1": {"sweep": {"points": 3.0}}}, "points", id="float-integer"),
        pytest.param({"Q1": {"sweep": {"start": 1}}}, "points", id="missing-required"),
        pytest.param(
            {"Q1": {"sweep": {"points": 3, "values": [1]}}}, "values", id="short-array"
        ),
        pytest.param(
            {"Q1": {"sweep": {"points": 3, "values": [1, {}]}}},
            "values",
            id="object-array-item",
        ),
    ],
)
def test_nested_recipe_tool_rejects_invalid_shapes_before_author_or_gui(
    tmp_path, overrides, path
):
    received: list[AuthorInputs] = []
    client = nested_client(tmp_path, received)
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
