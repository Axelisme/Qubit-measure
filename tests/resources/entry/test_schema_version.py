"""Whole-document component versions at the public entry read/write seam."""

from shlex import split

import pytest
from ruamel.yaml import YAML
from zcu_tools.resources.entry import (
    ResultEntry,
    component_registry,
    load_entry_template,
)
from zcu_tools.resources.parameter_schema import ParameterSchemaPolicy


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_runtime_rejects_missing_or_noncurrent_schema_without_writing(
    entry: ResultEntry, version: object
) -> None:
    path = entry.result_path / "setup.yaml"
    yaml = YAML()
    document = yaml.load(path)
    if version is None:
        document.pop("schema_version", None)
    else:
        document["schema_version"] = version
    yaml.dump(document, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="schema_version") as error:
        ResultEntry.open(
            "entry",
            result_root=entry.result_path.parent,
            database_root=entry.database_path.parent,
        )
    assert str(path) in str(error.value)
    assert "current" in str(error.value)
    assert "upgrade" in str(error.value)
    assert path.read_bytes() == original


@pytest.mark.parametrize("point", [None, "bias-a"])
def test_runtime_hint_targets_whole_entry(
    entry: ResultEntry, point: str | None
) -> None:
    if point is None:
        path = entry.result_path / "setup.yaml"
    else:
        entry.new_point(point)
        path = entry.result_path / "points" / point / "point.yaml"
    yaml = YAML()
    document = yaml.load(path)
    document.pop("schema_version")
    yaml.dump(document, path)

    if point is None:
        with pytest.raises(ValueError, match="schema_version missing") as error:
            ResultEntry.open(
                "entry",
                result_root=entry.result_path.parent,
                database_root=entry.database_path.parent,
            )
    else:
        with pytest.raises(ValueError, match="schema_version missing") as error:
            entry.use_point(point)
    preview, apply = str(error.value).splitlines()[-2:]
    assert split(preview.removeprefix("Preview: "))[-1] == str(entry.result_path)
    assert split(apply.removeprefix("Apply: "))[-2:] == [
        str(entry.result_path),
        "--apply",
    ]


def test_create_edit_seed_and_clone_use_caller_schema_version(
    entry: ResultEntry,
) -> None:
    before = component_registry.schema_policy
    component_registry.configure_schema(ParameterSchemaPolicy(4, ("caller-upgrade",)))
    try:
        created = ResultEntry.create(
            "current",
            result_root=entry.result_path.parent,
            database_root=entry.database_path.parent,
        )
        created.setup.add_component("S1", kind="fake/sensor")
        created.new_point("seed")
        created.new_point("clone", clone_from="seed")
        for path in (
            created.result_path / "setup.yaml",
            *created.result_path.glob("points/*/point.yaml"),
        ):
            document = YAML(typ="safe").load(path)
            assert document["schema_version"] == 4
            assert document["format_version"] == "2.0"
            assert "S1" in document["components"]
    finally:
        component_registry.configure_schema(before)


def test_template_requires_current_version_before_component_parsing(
    entry: ResultEntry,
) -> None:
    path = entry.result_path / "template.yaml"
    path.write_text(
        "format: zcu.entry-template\nformat_version: '1.0'\n"
        "components: {}\naliases:\n  format: zcu.aliases\n"
        "  format_version: '1.0'\n  aliases: {}\n",
        encoding="utf-8",
    )
    original = path.read_bytes()
    with pytest.raises(ValueError, match="schema_version missing") as error:
        load_entry_template(path)
    preview, apply = str(error.value).splitlines()[-2:]
    assert split(preview.removeprefix("Preview: "))[-1] == str(path)
    assert split(apply.removeprefix("Apply: "))[-2:] == [str(path), "--apply"]
    assert path.read_bytes() == original
