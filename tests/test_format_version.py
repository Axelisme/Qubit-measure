from collections.abc import Callable
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from zcu_tools.format_version import (
    FormatError,
    FormatVersion,
    MigrationError,
    MigrationRegistry,
    VersionError,
    YamlMap,
    YamlValue,
    validate_header,
)


def validate_synthetic_header(
    document: YamlMap, source: Path = Path("entry/setup.yaml")
) -> FormatVersion:
    return validate_header(
        document,
        expected_format="zcu.synthetic",
        supported_version=FormatVersion(1, 0),
        source=source,
    )


@pytest.mark.parametrize("minor", [0, 7])
def test_header_accepts_supported_and_newer_minor(minor: int) -> None:
    document: YamlMap = {
        "format": "zcu.synthetic",
        "format_version": f"1.{minor}",
        "future": {"nested": [1, "retained"]},
    }

    version = validate_synthetic_header(document)

    assert version == FormatVersion(1, minor)
    assert document["future"] == {"nested": [1, "retained"]}
    assert document["format_version"] == f"1.{minor}"


@pytest.mark.parametrize("actual_format", ["wrong.format", None])
def test_header_reports_format_mismatch(actual_format: str | None) -> None:
    source = Path("entry/setup.yaml")
    document: YamlMap = {"format": actual_format, "format_version": "1.0"}

    with pytest.raises(FormatError, match="format: expected") as caught:
        validate_synthetic_header(document, source)

    assert caught.value.source == source
    assert caught.value.field == "format"
    assert caught.value.actual == actual_format
    assert caught.value.expected == "zcu.synthetic"
    assert str(source) in str(caught.value)


@pytest.mark.parametrize("major", [0, 2])
def test_header_reports_unsupported_major(major: int) -> None:
    source = Path("entry/setup.yaml")
    document: YamlMap = {"format": "zcu.synthetic", "format_version": f"{major}.0"}

    with pytest.raises(VersionError) as caught:
        validate_synthetic_header(document, source)

    assert caught.value.source == source
    assert caught.value.field == "format_version"
    assert caught.value.actual == f"{major}.0"
    assert caught.value.expected == "1.0"
    assert str(source) in str(caught.value)


@pytest.mark.parametrize(
    "raw_version",
    [None, 1, [], "1", "1.0.0", "one.0", "1.-1", "1.2e3", "+1.0", "1.0\n"],
)
def test_header_rejects_malformed_version(raw_version: YamlValue) -> None:
    source = Path("entry/setup.yaml")
    document: YamlMap = {"format": "zcu.synthetic", "format_version": raw_version}

    with pytest.raises(VersionError) as caught:
        validate_synthetic_header(document, source)

    assert caught.value.source == source
    assert caught.value.field == "format_version"
    assert caught.value.actual == raw_version
    assert str(source) in str(caught.value)


@pytest.mark.parametrize("major, minor", [(-1, 0), (0, -1), (True, 0), (0, False)])
def test_version_components_are_non_negative_integers(major: int, minor: int) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        FormatVersion(major, minor)


def test_migration_same_version_returns_independent_document() -> None:
    document: YamlMap = {
        "format": "zcu.synthetic",
        "format_version": "1.7",
        "future": {"nested": [1, "retained"]},
    }

    result = MigrationRegistry().migrate(
        document,
        format="zcu.synthetic",
        target_version=FormatVersion(1, 7),
        source=Path("entry/setup.yaml"),
    )

    assert result == document
    future = result["future"]
    if not isinstance(future, dict):
        pytest.fail("future field must remain a mapping")
    future["nested"] = ["changed"]
    assert document["future"] == {"nested": [1, "retained"]}
    assert result["format_version"] == "1.7"


def test_migration_applies_registered_chain_across_major_without_mutating_input() -> (
    None
):
    registry = MigrationRegistry()
    document: YamlMap = {
        "format": "zcu.synthetic",
        "format_version": "1.0",
        "visited": [],
        "future": {"nested": [1, "retained"]},
    }

    def advance(doc: YamlMap, version: str) -> YamlMap:
        visited = doc["visited"]
        if not isinstance(visited, list):
            raise TypeError("visited must be a list")
        visited.append(version)
        doc["format_version"] = version
        return doc

    registry.register(
        "zcu.synthetic",
        FormatVersion(1, 0),
        FormatVersion(1, 1),
        lambda doc: advance(doc, "1.1"),
    )
    registry.register(
        "zcu.synthetic",
        FormatVersion(1, 1),
        FormatVersion(2, 0),
        lambda doc: advance(doc, "2.0"),
    )

    result = registry.migrate(
        document,
        format="zcu.synthetic",
        target_version=FormatVersion(2, 0),
        source=Path("entry/setup.yaml"),
    )

    assert result["format_version"] == "2.0"
    assert result["visited"] == ["1.1", "2.0"]
    assert result["future"] == {"nested": [1, "retained"]}
    assert document["format_version"] == "1.0"
    assert document["visited"] == []


def migration_to(version: FormatVersion) -> Callable[[YamlMap], YamlMap]:
    def step(document: YamlMap) -> YamlMap:
        document["format_version"] = f"{version.major}.{version.minor}"
        return document

    return step


@pytest.mark.parametrize(
    "existing, start, end",
    [
        (
            [(FormatVersion(1, 0), FormatVersion(1, 1))],
            FormatVersion(1, 0),
            FormatVersion(2, 0),
        ),
        ([], FormatVersion(2, 0), FormatVersion(1, 0)),
        ([], FormatVersion(1, 0), FormatVersion(1, 0)),
        (
            [(FormatVersion(1, 1), FormatVersion(2, 0))],
            FormatVersion(1, 0),
            FormatVersion(2, 0),
        ),
        (
            [(FormatVersion(1, 0), FormatVersion(2, 0))],
            FormatVersion(1, 1),
            FormatVersion(2, 0),
        ),
        (
            [(FormatVersion(1, 0), FormatVersion(2, 0))],
            FormatVersion(0, 0),
            FormatVersion(1, 1),
        ),
    ],
    ids=[
        "duplicate-start",
        "backward",
        "same-version",
        "skip-known",
        "insert-skipped",
        "insert-skipped-end",
    ],
)
def test_registry_rejects_invalid_chain_registration(
    existing: list[tuple[FormatVersion, FormatVersion]],
    start: FormatVersion,
    end: FormatVersion,
) -> None:
    registry = MigrationRegistry()
    for edge_start, edge_end in existing:
        registry.register("zcu.synthetic", edge_start, edge_end, migration_to(edge_end))

    with pytest.raises(MigrationError) as caught:
        registry.register("zcu.synthetic", start, end, migration_to(end))

    assert caught.value.format == "zcu.synthetic"
    assert caught.value.from_version == start
    assert caught.value.target_version == end
    assert caught.value.detail
    for edge_start, edge_end in existing:
        result = registry.migrate(
            {
                "format": "zcu.synthetic",
                "format_version": f"{edge_start.major}.{edge_start.minor}",
            },
            format="zcu.synthetic",
            target_version=edge_end,
            source=Path("entry/setup.yaml"),
        )
        assert result["format_version"] == f"{edge_end.major}.{edge_end.minor}"


@pytest.mark.parametrize(
    "edges, start, target",
    [
        ([], FormatVersion(1, 0), FormatVersion(2, 0)),
        (
            [(FormatVersion(1, 0), FormatVersion(1, 1))],
            FormatVersion(1, 0),
            FormatVersion(2, 0),
        ),
        (
            [(FormatVersion(1, 0), FormatVersion(2, 0))],
            FormatVersion(1, 0),
            FormatVersion(1, 1),
        ),
        ([], FormatVersion(2, 0), FormatVersion(1, 0)),
    ],
    ids=["no-chain", "gap", "overshoot", "downgrade"],
)
def test_migration_rejects_unreachable_target_without_mutating_input(
    edges: list[tuple[FormatVersion, FormatVersion]],
    start: FormatVersion,
    target: FormatVersion,
) -> None:
    registry = MigrationRegistry()
    for edge_start, edge_end in edges:
        registry.register("zcu.synthetic", edge_start, edge_end, migration_to(edge_end))
    raw_version = f"{start.major}.{start.minor}"
    document: YamlMap = {"format": "zcu.synthetic", "format_version": raw_version}

    with pytest.raises(MigrationError) as caught:
        registry.migrate(
            document,
            format="zcu.synthetic",
            target_version=target,
            source=Path("entry/setup.yaml"),
        )

    assert caught.value.format == "zcu.synthetic"
    assert caught.value.from_version == start
    assert caught.value.target_version == target
    assert caught.value.detail
    assert document["format_version"] == raw_version


@pytest.mark.parametrize(
    "output",
    [
        {"format": "wrong.format", "format_version": "1.1"},
        {"format": "zcu.synthetic", "format_version": "1.0"},
        {"format": "zcu.synthetic", "format_version": "1.2"},
        {"format": "zcu.synthetic", "format_version": "2.0"},
        {"format": "zcu.synthetic", "format_version": "bad"},
        {"format": "zcu.synthetic"},
    ],
    ids=[
        "format",
        "unchanged",
        "wrong-minor",
        "wrong-major",
        "malformed",
        "missing-version",
    ],
)
def test_migration_validates_each_step_before_next_step(output: YamlMap) -> None:
    registry = MigrationRegistry()
    document: YamlMap = {"format": "zcu.synthetic", "format_version": "1.0"}

    def invalid_step(doc: YamlMap) -> YamlMap:
        doc.clear()
        doc.update(output)
        return doc

    def forbidden_followup(doc: YamlMap) -> YamlMap:
        pytest.fail("invalid output must be rejected before the next step")

    registry.register(
        "zcu.synthetic", FormatVersion(1, 0), FormatVersion(1, 1), invalid_step
    )
    registry.register(
        "zcu.synthetic", FormatVersion(1, 1), FormatVersion(2, 0), forbidden_followup
    )

    with pytest.raises(MigrationError) as caught:
        registry.migrate(
            document,
            format="zcu.synthetic",
            target_version=FormatVersion(2, 0),
            source=Path("entry/setup.yaml"),
        )

    assert caught.value.format == "zcu.synthetic"
    assert caught.value.from_version == FormatVersion(1, 0)
    assert caught.value.target_version == FormatVersion(2, 0)
    assert caught.value.detail
    assert document == {"format": "zcu.synthetic", "format_version": "1.0"}


def test_migration_reports_step_failure_with_original_cause() -> None:
    registry = MigrationRegistry()
    document: YamlMap = {"format": "zcu.synthetic", "format_version": "1.0"}
    cause = ValueError("synthetic conversion failed")

    def broken_step(doc: YamlMap) -> YamlMap:
        doc["format_version"] = "1.1"
        raise cause

    registry.register(
        "zcu.synthetic", FormatVersion(1, 0), FormatVersion(1, 1), broken_step
    )

    with pytest.raises(MigrationError) as caught:
        registry.migrate(
            document,
            format="zcu.synthetic",
            target_version=FormatVersion(1, 1),
            source=Path("entry/setup.yaml"),
        )

    assert caught.value.__cause__ is cause
    assert caught.value.from_version == FormatVersion(1, 0)
    assert caught.value.target_version == FormatVersion(1, 1)
    assert "synthetic conversion failed" in caught.value.detail
    assert document["format_version"] == "1.0"


def test_yaml_writeback_preserves_newer_minor_unknown_fields_and_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.yaml"
    destination = tmp_path / "copy.yaml"
    original = (
        "format: zcu.synthetic\nformat_version: '1.7'\n"
        "future:\n  nested: [1, retained]\n"
    )
    source.write_text(original, encoding="utf-8")

    result = MigrationRegistry().migrate_yaml(
        source,
        destination,
        format="zcu.synthetic",
        target_version=FormatVersion(1, 7),
    )

    assert result == destination
    assert source.read_text(encoding="utf-8") == original
    assert YAML(typ="safe").load(destination) == {
        "format": "zcu.synthetic",
        "format_version": "1.7",
        "future": {"nested": [1, "retained"]},
    }
