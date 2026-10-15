"""Caller-owned named alias data for app composition, without storage or UI."""

from collections.abc import Mapping

from .aliases import AliasRow, AliasTable

# Preset names map to authored rows in table order; "default" supplies initial rows.
# None at app entrances means no catalog, distinct from an invalid empty mapping.
type AliasPresets = Mapping[str, tuple[AliasRow, ...]]


def validate_alias_presets(presets: AliasPresets | None) -> None:
    """Validate a caller catalog before app assembly, without I/O or mutation.

    None means no presets. A supplied mapping must contain "default", whose
    rows may be empty. Raise ValueError for a missing default (including {}),
    or InvalidInputError for any table's syntax, duplicate names or forward
    links. Each table uses the existing AliasTable contract; no component or
    device lookup occurs. Names and all authored rows remain caller data.
    """
    if presets is None:
        return
    if "default" not in presets:
        raise ValueError("Alias presets must contain 'default'")
    for rows in presets.values():
        AliasTable(rows)
