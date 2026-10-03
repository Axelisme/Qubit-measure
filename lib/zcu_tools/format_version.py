"""Shared artifact headers and explicit migrations; no legacy adapters."""

import re
from collections.abc import Callable, Mapping
from copy import deepcopy
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


class MigrationError(ValueError):
    def __init__(
        self,
        format: str,
        from_version: FormatVersion,
        target_version: FormatVersion,
        detail: str,
    ) -> None:
        self.format = format
        self.from_version = from_version
        self.target_version = target_version
        self.detail = detail
        super().__init__(f"{format}: {from_version} -> {target_version}: {detail}")


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


class MigrationRegistry:
    def __init__(self) -> None:
        self._steps: dict[
            tuple[str, FormatVersion],
            tuple[FormatVersion, Callable[[YamlMap], YamlMap]],
        ] = {}

    def register(
        self,
        format: str,
        from_version: FormatVersion,
        to_version: FormatVersion,
        step: Callable[[YamlMap], YamlMap],
    ) -> None:
        self._steps[format, from_version] = (to_version, step)

    def migrate(
        self,
        document: Mapping[str, YamlValue],
        *,
        format: str,
        target_version: FormatVersion,
        source: Path,
    ) -> YamlMap:
        version = _parse_version(document.get("format_version"), source)
        validate_header(
            document,
            expected_format=format,
            supported_version=version,
            source=source,
        )
        result = deepcopy(dict(document))
        while version != target_version:
            registered = self._steps.get((format, version))
            if registered is None:
                raise NotImplementedError("Migration chain is incomplete")
            next_version, step = registered
            result = step(result)
            version = next_version
        return result

    def migrate_yaml(
        self,
        source: Path,
        destination: Path,
        *,
        format: str,
        target_version: FormatVersion,
    ) -> Path:
        raise NotImplementedError((source, destination, format, target_version))
