"""Shared artifact headers; container versions do not govern component evolution."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

type YamlValue = (
    None | bool | int | float | str | list[YamlValue] | dict[str, YamlValue]
)
type YamlMap = dict[str, YamlValue]


@dataclass(frozen=True)
class FormatVersion:
    major: int
    minor: int

    def __post_init__(self) -> None:
        for field, value in (("major", self.major), ("minor", self.minor)):
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{field} must be a non-negative integer, got {value!r}"
                )


class FormatError(ValueError):
    def __init__(
        self, source: Path, field: str, actual: YamlValue, expected: YamlValue
    ) -> None:
        self.source = source
        self.field = field
        self.actual = actual
        self.expected = expected
        super().__init__(f"{source}: {field}: expected {expected!r}, got {actual!r}")


class VersionError(FormatError):
    pass


def _parse_version(raw_version: YamlValue, source: Path) -> FormatVersion:
    expected = "non-negative major.minor"
    if (
        not isinstance(raw_version, str)
        or re.fullmatch(r"[0-9]+\.[0-9]+", raw_version) is None
    ):
        raise VersionError(source, "format_version", raw_version, expected)
    major, minor = raw_version.split(".")
    try:
        return FormatVersion(int(major), int(minor))
    except ValueError as exc:
        raise VersionError(source, "format_version", raw_version, expected) from exc


def validate_header(
    document: Mapping[str, YamlValue],
    *,
    expected_format: str,
    supported_version: FormatVersion,
    source: Path,
) -> FormatVersion:
    actual_format = document.get("format")
    if actual_format != expected_format:
        raise FormatError(source, "format", actual_format, expected_format)
    raw_version = document.get("format_version")
    version = _parse_version(raw_version, source)
    if version.major != supported_version.major:
        raise VersionError(
            source,
            "format_version",
            raw_version,
            f"{supported_version.major}.{supported_version.minor}",
        )
    return version
