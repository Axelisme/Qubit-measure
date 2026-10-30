"""Text-budget contracts using synthetic wire inputs, without importing a lab."""

from __future__ import annotations

from check_mcp_text_budget import Overage, WireTool, check_text_budget


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
