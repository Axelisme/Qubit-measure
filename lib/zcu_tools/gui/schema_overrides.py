"""Helpers for applying safe, path-based schema overrides."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Optional, cast

from zcu_tools.gui.adapter import (
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ChannelValue,
    ModuleRefValue,
    ScalarSpec,
    ScalarValue,
    SweepValue,
    WaveformRefValue,
)


def _split_path(path: str) -> list[str]:
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise RuntimeError("Schema override path must not be empty")
    return parts


def _get_parent_section_spec(root: CfgSectionSpec, parts: list[str]) -> CfgSectionSpec:
    spec = root
    for part in parts[:-1]:
        next_spec = spec.fields.get(part)
        if not isinstance(next_spec, CfgSectionSpec):
            raise RuntimeError(
                f"Schema path '{'.'.join(parts)}' has no section '{part}'"
            )
        spec = next_spec
    return spec


def _get_parent_section_value(
    root: CfgSectionValue, parts: list[str]
) -> Optional[CfgSectionValue]:
    value = root
    for part in parts[:-1]:
        next_val = value.fields.get(part)
        if not isinstance(next_val, CfgSectionValue):
            return None
        value = next_val
    return value


def _coerce_value_for_node(current: CfgNodeValue, value: object) -> CfgNodeValue:
    if isinstance(current, ScalarValue):
        return ScalarValue(value=value, is_unset=False)
    if isinstance(current, ChannelValue):
        if not isinstance(value, (int, str)):
            raise RuntimeError(
                f"ChannelValue path requires int|str default, got {type(value).__name__}"
            )
        resolved = value if isinstance(value, int) else None
        return ChannelValue(chosen=value, resolved=resolved)
    if isinstance(current, SweepValue):
        if not isinstance(value, SweepValue):
            raise RuntimeError("SweepValue path requires SweepValue default")
        return value
    if isinstance(current, (CfgSectionValue, ModuleRefValue, WaveformRefValue)):
        if type(value) is not type(current):
            raise RuntimeError(
                f"Path requires {type(current).__name__} default, got {type(value).__name__}"
            )
        return cast(CfgNodeValue, value)
    raise RuntimeError(
        f"Unsupported node type for default override: {type(current).__name__}"
    )


def lock_field(schema: CfgSchema, path: str) -> CfgSchema:
    copied = copy.deepcopy(schema)
    parts = _split_path(path)
    parent = _get_parent_section_spec(copied.spec, parts)
    leaf = parent.fields.get(parts[-1])
    if not isinstance(leaf, ScalarSpec):
        raise RuntimeError(f"Path '{path}' does not point to a ScalarSpec")
    parent.fields[parts[-1]] = replace(leaf, editable=False)
    return copied


def hide_field(schema: CfgSchema, path: str) -> CfgSchema:
    copied = copy.deepcopy(schema)
    parts = _split_path(path)
    parent = _get_parent_section_spec(copied.spec, parts)
    leaf = parent.fields.get(parts[-1])
    if not isinstance(leaf, ScalarSpec):
        raise RuntimeError(f"Path '{path}' does not point to a ScalarSpec")
    parent.fields[parts[-1]] = replace(leaf, hidden=True)
    return copied


def set_default_value(schema: CfgSchema, path: str, value: object) -> CfgSchema:
    copied = copy.deepcopy(schema)
    parts = _split_path(path)
    parent = _get_parent_section_value(copied.value, parts)
    if parent is None or parts[-1] not in parent.fields:
        raise RuntimeError(f"Schema value path '{path}' does not exist")
    current = parent.fields[parts[-1]]
    parent.fields[parts[-1]] = _coerce_value_for_node(current, value)
    return copied


def set_field_label(schema: CfgSchema, path: str, label: str) -> CfgSchema:
    copied = copy.deepcopy(schema)
    parts = _split_path(path)
    parent = _get_parent_section_spec(copied.spec, parts)
    leaf = parent.fields.get(parts[-1])
    if leaf is None or not hasattr(leaf, "label"):
        raise RuntimeError(f"Path '{path}' does not support labels")
    parent.fields[parts[-1]] = replace(leaf, label=label)
    return copied


def set_field_choices(schema: CfgSchema, path: str, choices: list[object]) -> CfgSchema:
    copied = copy.deepcopy(schema)
    parts = _split_path(path)
    parent = _get_parent_section_spec(copied.spec, parts)
    leaf = parent.fields.get(parts[-1])
    if not isinstance(leaf, ScalarSpec):
        raise RuntimeError(f"Path '{path}' does not point to a ScalarSpec")
    parent.fields[parts[-1]] = replace(leaf, choices=choices)
    return copied


def apply_schema_overrides(
    schema: CfgSchema,
    *,
    spec_overrides: Optional[dict[str, dict[str, object]]] = None,
    value_overrides: Optional[dict[str, object]] = None,
) -> CfgSchema:
    updated = schema
    for path, overrides in (spec_overrides or {}).items():
        for key, value in overrides.items():
            if key == "editable":
                if value is False:
                    updated = lock_field(updated, path)
                elif value is not True:
                    raise RuntimeError(
                        f"Unsupported editable override at '{path}': {value!r}"
                    )
            elif key == "hidden":
                if value is True:
                    updated = hide_field(updated, path)
                elif value is not False:
                    raise RuntimeError(
                        f"Unsupported hidden override at '{path}': {value!r}"
                    )
            elif key == "label":
                if not isinstance(value, str):
                    raise RuntimeError(f"Label override at '{path}' must be str")
                updated = set_field_label(updated, path, value)
            elif key == "choices":
                if not isinstance(value, list):
                    raise RuntimeError(f"Choices override at '{path}' must be list")
                updated = set_field_choices(updated, path, value)
            else:
                raise RuntimeError(f"Unsupported spec override key '{key}' at '{path}'")

    for path, value in (value_overrides or {}).items():
        updated = set_default_value(updated, path, value)
    return updated
