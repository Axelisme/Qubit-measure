"""Cfg-form seam — the SINGLE allowed import point of the measure-app *cfg form
widget* machinery (``gui.app.main.ui.cfg_form`` + ``gui.app.main.live_model``).

The sibling ``cfg/__init__`` seam re-exports the pure spec/value data model;
this one re-exports the reactive Qt form that renders it: ``CfgFormWidget`` (the
viewer that attaches to a LiveModel) + ``SectionLiveField`` / ``ScalarLiveField``
/ ``LiveModelEnv`` (the LiveModel layer it renders). autofluxdep's node detail
pane builds a ``SectionLiveField`` over a placement's ``NodeCfgSchema`` value
tree, wraps it in a ``CfgFormWidget``, and writes edits back through the
controller — exactly the local-draft pattern measure's inspect / writeback
dialogs use. App-shell scalar controls, such as the global flux sweep, reuse the
same ``ScalarLiveField`` + ``ScalarWidget`` pair so expression mode stays on the
shared cfg-widget path.

Keeping the widget import here (and only here) confines the app-to-app coupling
to one file: a future lift of the cfg form into a shared layer only retargets
this seam, not the node detail pane.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol, cast

from zcu_tools.gui.app.main.adapter import (
    CfgNodeSpec,
    CfgNodeValue,
    CfgSectionSpec,
    DeviceRefSpec,
    LiteralSpec,
    ModuleRefSpec,
    ScalarSpec,
    SweepSpec,
    WaveformRefSpec,
)
from zcu_tools.gui.app.main.live_model import (
    DeviceRefLiveField,
    LiveModelEnv,
    ModuleRefLiveField,
    ScalarLiveField,
    SectionLiveField,
    SweepLiveField,
)
from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget as _SharedCfgFormWidget
from zcu_tools.gui.app.main.ui.fields.common import ScalarWidget

Tone = Literal["normal", "muted", "info", "warning", "error"]


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


class CfgFormWidget(_SharedCfgFormWidget):
    """Autofluxdep cfg form seam with generic field decoration metadata.

    The shared renderer remains unchanged in this scope; this subclass computes the
    stable path -> decoration table that an app-specific presenter can consume.
    """

    def __init__(
        self,
        parent=None,
        *,
        field_label_max_width: int | None = None,
        decoration_provider: FieldDecorationProvider | None = None,
    ) -> None:
        super().__init__(parent, field_label_max_width=field_label_max_width)
        self._decoration_provider = decoration_provider
        self._field_decorations: dict[str, FieldDecoration] = {}

    def attach(self, model: SectionLiveField) -> None:
        self._field_decorations = _collect_decorations(model, self._decoration_provider)
        super().attach(model)

    def set_decoration_provider(self, provider: FieldDecorationProvider | None) -> None:
        self._decoration_provider = provider
        model = self.get_live_root()
        self._field_decorations = (
            {} if model is None else _collect_decorations(model, provider)
        )

    def decoration_for_path(self, path: str) -> FieldDecoration:
        try:
            return self._field_decorations[path]
        except KeyError as exc:
            raise KeyError(f"Unknown cfg field path {path!r}") from exc

    def decoration_paths(self) -> tuple[str, ...]:
        return tuple(self._field_decorations)

    def _emit_schema_changed(self, *_: object) -> None:
        model = self.get_live_root()
        if model is not None:
            self._field_decorations = _collect_decorations(
                model, self._decoration_provider
            )
        super()._emit_schema_changed(*_)


def default_decoration_for_spec(spec: CfgNodeSpec) -> FieldDecoration:
    """Return the generic decoration implied by a pure cfg spec."""
    if isinstance(spec, LiteralSpec):
        return FieldDecoration(hidden=True, enabled=False)
    if isinstance(spec, (ScalarSpec, SweepSpec)):
        return FieldDecoration(enabled=bool(spec.editable))
    if isinstance(
        spec, (ModuleRefSpec, WaveformRefSpec, DeviceRefSpec, CfgSectionSpec)
    ):
        return FieldDecoration(enabled=True)
    raise TypeError(f"Unsupported cfg spec type {type(spec).__name__}")


def _collect_decorations(
    root: SectionLiveField,
    provider: FieldDecorationProvider | None,
) -> dict[str, FieldDecoration]:
    decorations: dict[str, FieldDecoration] = {}
    _collect_section_decorations("", root, provider, decorations)
    return decorations


def _collect_section_decorations(
    base_path: str,
    field: SectionLiveField,
    provider: FieldDecorationProvider | None,
    decorations: dict[str, FieldDecoration],
) -> None:
    for key, child in field.fields.items():
        path = f"{base_path}.{key}" if base_path else key
        _collect_field_decoration(path, child, provider, decorations)


def _collect_field_decoration(
    path: str,
    field: object,
    provider: FieldDecorationProvider | None,
    decorations: dict[str, FieldDecoration],
) -> None:
    spec = cast(CfgNodeSpec, getattr(field, "spec"))
    value = cast(CfgNodeValue | None, field.get_value())  # type: ignore[attr-defined]
    patch = None if provider is None else provider.decoration_for(path, spec, value)
    decorations[path] = default_decoration_for_spec(spec).merge(patch)
    if isinstance(field, SectionLiveField):
        _collect_section_decorations(path, field, provider, decorations)
        return
    if isinstance(field, ModuleRefLiveField) and field.sub_field is not None:
        _collect_section_decorations(path, field.sub_field, provider, decorations)
        return
    if isinstance(field, SweepLiveField):
        return
    if isinstance(field, (ScalarLiveField, DeviceRefLiveField)):
        return


__all__ = [
    "CfgFormWidget",
    "FieldDecoration",
    "FieldDecorationPatch",
    "FieldDecorationProvider",
    "LiveModelEnv",
    "ScalarLiveField",
    "ScalarWidget",
    "SectionLiveField",
    "Tone",
    "default_decoration_for_spec",
]
