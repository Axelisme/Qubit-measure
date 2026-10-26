"""Portable entry templates validate at the resources boundary."""

from pathlib import Path

import pytest
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap


@pytest.fixture
def template_document() -> YamlMap:
    return {
        "format": "zcu.entry-template",
        "format_version": "1.0",
        "schema_version": 1,
        "components": {
            "D1": {"kind": "fake/drive/a", "rate": 123.0, "sense": "S1"},
            "S1": {"kind": "fake/sensor", "rate": 456.0},
        },
        "aliases": {
            "format": "zcu.aliases",
            "format_version": "1.0",
            "aliases": {"drive": "D1", "sense": "$drive.sense", "bias": ""},
        },
    }


def test_reader_preserves_models_defaults_and_order(
    tmp_path: Path, template_document: YamlMap
) -> None:
    from zcu_tools.resources.entry import load_entry_template

    path = tmp_path / "template.yaml"
    YAML().dump(template_document, path)
    template = load_entry_template(path)

    assert list(template.components) == ["D1", "S1"]
    assert template.components["D1"].model_dump()["rate"] == 123.0
    assert template.components["D1"].model_dump()["ext"] == {}
    assert list(template.aliases.aliases.items()) == [
        ("drive", "D1"),
        ("sense", "$drive.sense"),
        ("bias", ""),
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "other", "format"),
        ("format_version", "2.0", "format_version"),
        ("components", {"D1": {"kind": "not/registered"}}, "unknown kind"),
        ("components", {"D1": {"kind": "fake/drive/a", "typo": 1}}, "typo"),
        ("components", {"D1": {"kind": "fake/drive/a", "rate": "bad"}}, "rate"),
        ("components", {"_hidden": {"kind": "fake/drive/a"}}, "component name"),
        ("components", {"D1": {}}, "kind"),
        (
            "aliases",
            {"format": "wrong", "format_version": "1.0", "aliases": {}},
            "format",
        ),
        (
            "aliases",
            {"format": "zcu.aliases", "format_version": "2.0", "aliases": {}},
            "format_version",
        ),
        (
            "aliases",
            {
                "format": "zcu.aliases",
                "format_version": "1.0",
                "aliases": {"drive": None},
            },
            "string",
        ),
    ],
)
def test_reader_rejects_invalid_documents_at_load(
    tmp_path: Path,
    template_document: YamlMap,
    field: str,
    value: YamlMap | str,
    message: str,
) -> None:
    from zcu_tools.resources.entry import load_entry_template

    template_document[field] = value
    path = tmp_path / "invalid.yaml"
    YAML().dump(template_document, path)
    with pytest.raises(ValueError, match=message):
        load_entry_template(path)
