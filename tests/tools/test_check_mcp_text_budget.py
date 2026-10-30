"""Text-budget contracts using synthetic wire inputs, without importing a lab."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable

import pytest
from check_mcp_text_budget import (
    Overage,
    WireTool,
    check_text_budget,
    collect_text_budget,
)


@pytest.fixture
def wire_reply(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    def configure(stdout: str) -> None:
        def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)

        monkeypatch.setattr(subprocess, "run", run)

    return configure


@pytest.mark.parametrize(
    "stdout",
    [
        "[]",
        "false",
        "null",
        '"not a response"',
        '{"id": true, "result": {}}\n{"id": 2, "result": {"tools": []}}',
        '{"id": 1.0, "result": {}}\n{"id": 2, "result": {"tools": []}}',
        '{"id": 1, "result": []}\n{"id": 2, "result": {"tools": []}}',
        '{"id": 1, "result": null}',
        '{"id": 1}',
        '{"id": 1, "error": {"message": "failed"}}',
        '{"id": 2, "result": {"tools": []}}',
        '{"id": 1, "result": {}}',
        '{"id": 1, "result": {}}\n{"id": 2, "result": {}}',
        '{"id": 1, "result": {}}\n{"id": 2, "result": {"tools": {}}}',
        '{"id": 1, "result": {}}\n{"id": 2, "result": {"tools": null}}',
    ],
)
def test_collection_rejects_malformed_required_wire_envelopes(
    wire_reply: Callable[[str], None], stdout: str
) -> None:
    wire_reply(stdout)

    with pytest.raises(ValueError, match="MCP"):
        collect_text_budget(set())


def test_collection_measures_decoded_wire_with_exact_recipe_names(
    wire_reply: Callable[[str], None],
) -> None:
    wire_reply(
        '{"id": 1, "result": {"instructions": "\\u91cf"}}\n'
        + json.dumps(
            {
                "id": 2,
                "result": {
                    "tools": [
                        {"name": "member", "description": "量" * 201, "inputSchema": {}}
                    ]
                },
            }
        )
    )

    report = collect_text_budget({"member"})

    assert report.instructions_length == 1
    assert report.overages == (Overage("tools/member/description", 201, 200),)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_collection_rejects_invalid_timeout_before_launch(
    monkeypatch: pytest.MonkeyPatch, timeout: float
) -> None:
    def unexpected_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid timeout must fail before subprocess launch")

    monkeypatch.setattr(subprocess, "run", unexpected_launch)

    with pytest.raises(ValueError, match="timeout"):
        collect_text_budget(set(), timeout=timeout)


@pytest.mark.parametrize("text", ["false", "[]", '["not text"]', "0", "{}"])
def test_rejects_non_text_instructions(text: str) -> None:
    with pytest.raises(ValueError, match="instructions"):
        check_text_budget(json.loads(text), [], set())


@pytest.mark.parametrize(
    "item",
    [
        "false",
        "[]",
        "{}",
        '{"name": false, "inputSchema": {}}',
        '{"name": [], "inputSchema": {}}',
        '{"name": "tool"}',
        '{"name": "tool", "inputSchema": null}',
        '{"name": "tool", "inputSchema": []}',
        '{"name": "tool", "inputSchema": false}',
        '{"name": "tool", "inputSchema": {}, "description": false}',
        '{"name": "tool", "inputSchema": {}, "description": ["not text"]}',
    ],
)
def test_rejects_invalid_tool_text_and_required_structure(item: str) -> None:
    with pytest.raises(ValueError, match="tools"):
        check_text_budget(None, [json.loads(item)], set())


@pytest.mark.parametrize(
    "schema",
    [
        '{"description": false}',
        '{"description": ["not text"]}',
        '{"properties": {"field": {"description": false}}}',
        '{"properties": {"field": {"description": ["not text"]}}}',
        '{"properties": {"field": "not a schema"}}',
        '{"$defs": {"Shared": null}}',
        '{"definitions": {"Shared": []}}',
        '{"patternProperties": {".*": 0}}',
        '{"dependentSchemas": {"field": "not a schema"}}',
        '{"items": null}',
        '{"additionalProperties": []}',
        '{"anyOf": ["not a schema"]}',
        '{"prefixItems": [null]}',
        '{"allOf": {}}',
        '{"properties": []}',
        '{"oneOf": null}',
    ],
)
def test_rejects_invalid_descriptions_and_visited_schema_structure(schema: str) -> None:
    tool: WireTool = {"name": "tool", "inputSchema": json.loads(schema)}

    with pytest.raises(ValueError, match="tools/tool/inputSchema"):
        check_text_budget(None, [tool], set())


def test_boolean_subschemas_and_instance_data_are_not_rejected_or_measured() -> None:
    tool: WireTool = {
        "name": "tool",
        "inputSchema": {
            "description": None,
            "properties": {"yes": True, "no": False, "empty": {"description": None}},
            "$defs": {"Any": True, "Never": False},
            "additionalProperties": False,
            "items": True,
            "anyOf": [True, False],
            "default": {"description": False, "properties": []},
            "examples": [{"description": ["instance data"]}],
            "const": {"items": None},
            "enum": [{"description": False}],
        },
    }

    report = check_text_budget(None, [tool], set())

    assert report.exit_code == 0
    assert report.overages == ()


def test_inclusive_budgets_count_characters_and_allow_absent_descriptions() -> None:
    tools: list[WireTool] = [
        {
            "name": "catalog_member",
            "description": "量" * 200,
            "inputSchema": {"properties": {"field": {"description": "量" * 100}}},
        },
        {
            "name": "recipe_not_in_catalog",
            "description": "a" * 400,
            "inputSchema": {},
        },
        {"name": "undocumented", "inputSchema": {}},
        {"name": "nullable", "description": None, "inputSchema": {}},
    ]

    report = check_text_budget("量" * 2000, tools, {"catalog_member"})

    assert report.exit_code == 0
    assert report.overages == ()
    assert report.instructions_length == 2000
    assert report.tool_count == 4
    assert check_text_budget(None, [], set()).exit_code == 0


def test_reports_every_excess_with_its_length_and_catalog_specific_limit() -> None:
    tools: list[WireTool] = [
        {
            "name": "catalog_member",
            "description": "量" * 201,
            "inputSchema": {"properties": {"field": {"description": "量" * 101}}},
        },
        {
            "name": "recipe_not_in_catalog",
            "description": "a" * 401,
            "inputSchema": {"properties": {"other": {"description": "b" * 102}}},
        },
    ]

    report = check_text_budget("量" * 2001, tools, {"catalog_member"})

    assert report.exit_code == 1
    assert report.overages == (
        Overage("instructions", 2001, 2000),
        Overage("tools/catalog_member/description", 201, 200),
        Overage(
            "tools/catalog_member/inputSchema/properties/field/description", 101, 100
        ),
        Overage("tools/recipe_not_in_catalog/description", 401, 400),
        Overage(
            "tools/recipe_not_in_catalog/inputSchema/properties/other/description",
            102,
            100,
        ),
    )


def test_nested_schema_properties_are_measured_without_counting_annotation_data() -> (
    None
):
    tools: list[WireTool] = [
        {
            "name": "nested",
            "inputSchema": {
                "description": "root context" * 100,
                "$defs": {
                    "Shared": {
                        "description": "type context" * 100,
                        "properties": {"value": {"description": "a" * 103}},
                    }
                },
                "properties": {
                    "edits": {
                        "items": {
                            "anyOf": [
                                {"properties": {"value": {"description": "b" * 104}}},
                                {"$ref": "#/$defs/Shared"},
                            ]
                        }
                    },
                    "settings": {
                        "properties": {"enabled": {"description": "c" * 105}},
                        "default": {"properties": {"data": {"description": "d" * 200}}},
                        "examples": [
                            {"properties": {"data": {"description": "e" * 200}}}
                        ],
                    },
                },
            },
        }
    ]

    report = check_text_budget(None, tools, set())

    assert report.exit_code == 1
    assert set(report.overages) == {
        Overage(
            "tools/nested/inputSchema/$defs/Shared/properties/value/description",
            103,
            100,
        ),
        Overage(
            "tools/nested/inputSchema/properties/edits/items/anyOf/0/properties/value/description",
            104,
            100,
        ),
        Overage(
            "tools/nested/inputSchema/properties/settings/properties/enabled/description",
            105,
            100,
        ),
    }
