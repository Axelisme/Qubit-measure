"""Presentation-only decoration contract for shared cfg widgets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol

from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CfgNodeSpec,
    CfgNodeValue,
    CfgSectionSpec,
    LiteralSpec,
    ReferenceSpec,
    ScalarSpec,
    SweepSpec,
)

Tone = Literal["normal", "muted", "info", "warning", "error"]


class FieldDecorationProtocol(Protocol):
    """Typed presentation surface consumed by field widgets."""

    @property
    def hidden(self) -> bool: ...

    @property
    def enabled(self) -> bool: ...

    @property
    def tone(self) -> str: ...

    @property
    def badge(self) -> str: ...

    @property
    def tooltip(self) -> str: ...

    @property
    def label_suffix(self) -> str: ...


@dataclass(frozen=True)
class FieldDecoration:
    """Presentation contract for one rendered cfg field path."""

    hidden: bool = False
    enabled: bool = True
    tone: Tone = "normal"
    badge: str = ""
    tooltip: str = ""
    label_suffix: str = ""

    def merge(self, patch: FieldDecorationPatch | None) -> FieldDecoration:
        if patch is None:
            return self
        updates = {
            name: value
            for name, value in {
                "hidden": patch.hidden,
                "enabled": patch.enabled,
                "tone": patch.tone,
                "badge": patch.badge,
                "tooltip": patch.tooltip,
                "label_suffix": patch.label_suffix,
            }.items()
            if value is not None
        }
        return replace(self, **updates)


@dataclass(frozen=True)
class FieldDecorationPatch:
    """Sparse app-specific patch over a field's default decoration."""

    hidden: bool | None = None
    enabled: bool | None = None
    tone: Tone | None = None
    badge: str | None = None
    tooltip: str | None = None
    label_suffix: str | None = None


class FieldDecorationProvider(Protocol):
    """App-specific decoration lookup keyed by full value-tree path."""

    def decoration_for(
        self,
        path: str,
        spec: CfgNodeSpec,
        value: CfgNodeValue | None,
    ) -> FieldDecorationPatch | None: ...


def default_decoration_for_spec(spec: CfgNodeSpec) -> FieldDecoration:
    """Return the generic decoration implied by a pure cfg spec."""
    if isinstance(spec, LiteralSpec):
        return FieldDecoration(hidden=True, enabled=False)
    if isinstance(spec, (ScalarSpec, SweepSpec, CenteredSweepSpec)):
        return FieldDecoration(enabled=bool(spec.editable), tooltip=spec.tooltip)
    if isinstance(spec, (ReferenceSpec, CfgSectionSpec)):
        return FieldDecoration(enabled=True)
    raise TypeError(f"Unsupported cfg spec type {type(spec).__name__}")


__all__ = [
    "FieldDecoration",
    "FieldDecorationPatch",
    "FieldDecorationProtocol",
    "FieldDecorationProvider",
    "Tone",
    "default_decoration_for_spec",
]
