"""Shared artifact headers and explicit migrations; no legacy adapters."""

import errno
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter
from ruamel.yaml import YAML

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
        if (format, from_version) in self._steps:
            raise MigrationError(
                format, from_version, to_version, "duplicate starting version"
            )
        start = (from_version.major, from_version.minor)
        end = (to_version.major, to_version.minor)
        if start >= end:
            raise MigrationError(
                format, from_version, to_version, "step must advance the version"
            )
        for (registered_format, edge_from), (edge_to, _) in self._steps.items():
            if registered_format != format:
                continue
            edge_start = (edge_from.major, edge_from.minor)
            edge_end = (edge_to.major, edge_to.minor)
            if (
                start < edge_start < end
                or start < edge_end < end
                or edge_start < start < edge_end
                or edge_start < end < edge_end
            ):
                raise MigrationError(
                    format,
                    from_version,
                    to_version,
                    "step skips a registered intermediate version",
                )
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
        from_version = version
        target = (target_version.major, target_version.minor)
        if (version.major, version.minor) > target:
            raise MigrationError(
                format, from_version, target_version, "cannot downgrade"
            )
        result = deepcopy(dict(document))
        while version != target_version:
            registered = self._steps.get((format, version))
            if registered is None:
                raise MigrationError(
                    format, from_version, target_version, f"missing step from {version}"
                )
            next_version, step = registered
            if (next_version.major, next_version.minor) > target:
                raise MigrationError(
                    format,
                    from_version,
                    target_version,
                    f"step from {version} overshoots target",
                )
            try:
                result = step(result)
            except Exception as exc:
                raise MigrationError(
                    format,
                    from_version,
                    target_version,
                    f"step from {version} to {next_version} failed: {exc}",
                ) from exc
            try:
                actual_version = validate_header(
                    result,
                    expected_format=format,
                    supported_version=next_version,
                    source=source,
                )
                if actual_version != next_version:
                    raise VersionError(
                        source,
                        "format_version",
                        result.get("format_version"),
                        f"{next_version.major}.{next_version.minor}",
                    )
            except FormatError as exc:
                raise MigrationError(
                    format,
                    from_version,
                    target_version,
                    f"invalid output from {version} to {next_version}: {exc}",
                ) from exc
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
        if (
            source.resolve() == destination.resolve()
            or destination.exists()
            or destination.is_symlink()
        ):
            raise FileExistsError(
                errno.EEXIST, "destination must be a new file", str(destination)
            )
        yaml = YAML(typ="safe")
        with source.open("r", encoding="utf-8") as stream:
            document = TypeAdapter(YamlMap).validate_python(
                yaml.load(stream), strict=True
            )
        result = self.migrate(
            document, format=format, target_version=target_version, source=source
        )
        with destination.open("x", encoding="utf-8") as stream:
            yaml.dump(result, stream)
        return destination
