"""Flat mutation-target projection, separate from complete cfg observations."""

from __future__ import annotations

from collections.abc import Iterable

from zcu_tools.gui.cfg import DirectValue, EvalValue, encode_complex
from zcu_tools.gui.cfg.binding import CfgDraft, SettableTarget, SettableTargetKind


def project_target_entries(draft: CfgDraft) -> list[dict[str, object]]:
    """Project live binding targets into the existing flat wire entry shape."""
    return project_targets(draft.iter_settable_targets())


def project_targets(targets: Iterable[SettableTarget]) -> list[dict[str, object]]:
    """Project an already-selected target snapshot without walking a draft."""
    return [_target_entry(target) for target in targets]


def _target_entry(target: SettableTarget) -> dict[str, object]:
    kind = (
        "moduleref_key"
        if target.kind is SettableTargetKind.REFERENCE_KEY
        else target.kind.value
    )
    entry: dict[str, object] = {
        "path": target.path,
        "kind": kind,
        "value": _wire_value(target.get_value()),
        "type": _wire_type(target.value_type),
    }
    choices = target.choices()
    if choices is not None:
        entry["choices"] = list(choices)
    return entry


def _wire_value(value: object) -> object:
    if isinstance(value, EvalValue):
        return value.expr
    if isinstance(value, DirectValue):
        return (
            encode_complex(value.value)
            if isinstance(value.value, complex)
            else value.value
        )
    return value


def _wire_type(value_type: type) -> str:
    if value_type is int:
        return "integer"
    if value_type is float:
        return "number"
    if value_type is str:
        return "string"
    if value_type is bool:
        return "bool"
    return value_type.__name__


__all__ = ["project_target_entries", "project_targets"]
