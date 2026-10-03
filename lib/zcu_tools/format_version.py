"""Shared artifact headers and explicit migrations; no legacy adapters."""

from collections.abc import Callable, Mapping
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
    if not isinstance(raw_version, str):
        raise NotImplementedError((document, expected_format, source))
    major, minor = raw_version.split(".")
    version = FormatVersion(int(major), int(minor))
    if version.major != supported_version.major:
        raise NotImplementedError((version, supported_version, source))
    return version


class MigrationRegistry:
    def register(
        self,
        format: str,
        from_version: FormatVersion,
        to_version: FormatVersion,
        step: Callable[[YamlMap], YamlMap],
    ) -> None:
        raise NotImplementedError((format, from_version, to_version, step))

    def migrate(
        self,
        document: Mapping[str, YamlValue],
        *,
        format: str,
        target_version: FormatVersion,
        source: Path,
    ) -> YamlMap:
        validate_header(
            document,
            expected_format=format,
            supported_version=target_version,
            source=source,
        )
        raise NotImplementedError("Migration chain is not implemented")

    def migrate_yaml(
        self,
        source: Path,
        destination: Path,
        *,
        format: str,
        target_version: FormatVersion,
    ) -> Path:
        raise NotImplementedError((source, destination, format, target_version))
