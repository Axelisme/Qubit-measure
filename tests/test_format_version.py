from pathlib import Path

import pytest
from zcu_tools.format_version import (
    FormatError,
    FormatVersion,
    VersionError,
    YamlMap,
    validate_header,
)


@pytest.mark.parametrize("minor", [0, 7])
def test_header_accepts_supported_and_newer_minor(minor: int) -> None:
    document: YamlMap = {
        "format": "zcu.synthetic",
        "format_version": f"1.{minor}",
        "future": {"nested": [1, "retained"]},
    }

    version = validate_header(
        document,
        expected_format="zcu.synthetic",
        supported_version=FormatVersion(1, 0),
        source=Path("entry/setup.yaml"),
    )

    assert version == FormatVersion(1, minor)
    assert document["future"] == {"nested": [1, "retained"]}
    assert document["format_version"] == f"1.{minor}"


@pytest.mark.parametrize("actual_format", ["wrong.format", None])
def test_header_reports_format_mismatch(actual_format: str | None) -> None:
    source = Path("entry/setup.yaml")
    document: YamlMap = {"format": actual_format, "format_version": "1.0"}

    with pytest.raises(FormatError, match="format: expected") as caught:
        validate_header(
            document,
            expected_format="zcu.synthetic",
            supported_version=FormatVersion(1, 0),
            source=source,
        )

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
        validate_header(
            document,
            expected_format="zcu.synthetic",
            supported_version=FormatVersion(1, 0),
            source=source,
        )

    assert caught.value.source == source
    assert caught.value.field == "format_version"
    assert caught.value.actual == f"{major}.0"
    assert caught.value.expected == "1.0"
    assert str(source) in str(caught.value)
