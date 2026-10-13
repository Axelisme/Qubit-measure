"""Opaque parameter metadata and the parameter-container header contract."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from pydantic import TypeAdapter

from zcu_tools.format_version import FormatVersion, YamlMap, YamlValue, validate_header

PARAMETER_FORMAT = "zcu.parameter-container"
PARAMETER_VERSION = FormatVersion(2, 0)


def validate_parameter_header(
    document: Mapping[str, YamlValue], *, source: Path
) -> FormatVersion:
    """Require a parameter-container 2.x header and reject the old source table.

    source is the document path for diagnostics. A wrong format or major raises
    the shared FormatError/VersionError. Any top-level provenance key raises
    ValueError, even in a future minor. Diagnostics include source, the actual
    version and expected version 2.0. No data is changed or migrated.
    """
    version = validate_header(
        document,
        expected_format=PARAMETER_FORMAT,
        supported_version=PARAMETER_VERSION,
        source=source,
    )
    if "provenance" in document:
        raise ValueError(
            f"{source}: format_version {document.get('format_version')!r}: "
            "expected 2.0 with metadata, not a provenance table"
        )
    return version


def parameter_path(path: str) -> tuple[str, ...]:
    """Split a nonempty dotted logical path, rejecting empty segments.

    Paths name caller fields or mapping keys, not YAML envelope fields. No
    escaping or list indexing is provided. Invalid paths raise ValueError;
    literal keys containing dots remain accessible through field item access.
    This function checks syntax only, not whether a value exists.
    """
    parts = tuple(path.split("."))
    if not all(parts):
        raise ValueError(f"{path!r}: expected nonempty dotted field segments")
    return parts


def read_metadata(table: Mapping[str, YamlMap], path: str) -> YamlMap | None:
    """Return a detached mapping for a logical path, or None when absent.

    table is the validated metadata side table. path has nonempty dotted
    segments. This memory-only read neither checks values nor interprets keys.
    An explicitly stored empty mapping is returned as {}, not None.
    """
    parameter_path(path)
    value = table.get(path)
    return deepcopy(value) if value is not None else None


def write_metadata(table: dict[str, YamlMap], path: str, metadata: object) -> None:
    """Replace one path's entire mapping in a caller-owned draft side table.

    metadata is an untrusted string-keyed YAML mapping of null, bool, int,
    float, str, lists and nested mappings. Strict representation validation
    raises Pydantic ValidationError for other types, without changing table.
    An empty mapping clears the path's content. No deep merge, domain
    validation, value change, source lookup or automatic key generation occurs.
    The copied input does not alias the caller. DocumentStore owns commit-time
    merging and conflicts; replacing a draft mapping does not change that unit.
    """
    parameter_path(path)
    accepted = TypeAdapter(YamlMap).validate_python(metadata, strict=True)
    table[path] = deepcopy(accepted)
