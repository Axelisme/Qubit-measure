"""CfgFormWidget — renders a CfgSchema as an interactive reactive Qt form.

Uses CfgDraft as the active data layer and delegates field rendering to
lib/zcu_tools/gui/ui/fields/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol, cast

from qtpy.QtCore import QTimer, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.main.adapter import (
    CenteredSweepSpec,
    CfgNodeSpec,
    CfgNodeValue,
    CfgSectionSpec,
    ChoiceSectionSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ScalarSpec,
    SweepSpec,
)
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    CfgDraft,
    CfgField,
    ReferenceField,
    ScalarField,
    SectionField,
    SweepField,
)

from .fields import SectionWidget, TextInputEnhancer

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import CfgSchema, CfgSectionValue

logger = logging.getLogger(__name__)

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


class CfgFormWidget(QWidget):
    """A pluggable viewer over a service-owned CfgDraft (ADR-0008).

    The widget does *not* own a CfgDraft — it ``attach``es to one that the
    ``CfgEditorService`` owns (renders it + reflects its changes) and
    ``detach``es without tearing it down. This makes an agent edit and a user
    view converge on the same model (WYSIWYG), and lets a model outlive the
    widget. md/ml-change refresh of EvalValue snapshots is the service's job (it
    owns the model); the widget repaints for free via the model's ``on_change``.

    Whether changes propagate to ``State.cfg_schema`` depends on who handles
    ``schema_changed``: in the tab pane it is bound to ``update_tab_cfg``
    (auto-commit); in inspect / writeback dialogs the host commits on Apply.
    """

    validity_changed: Signal = Signal(bool)
    # Auto-commit signal (in tab mode) / draft-change notification (in dialog
    # mode). Payload is a freshly built CfgSchema snapshot of the LiveModel.
    schema_changed: Signal = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        field_label_max_width: int | None = None,
        decoration_provider: FieldDecorationProvider | None = None,
        text_input_enhancer: TextInputEnhancer | None = None,
    ) -> None:
        super().__init__(parent)
        self._draft: CfgDraft | None = None
        self._root_widget: SectionWidget | None = None
        self._field_label_max_width = field_label_max_width
        self._decoration_provider = decoration_provider
        self._text_input_enhancer = text_input_enhancer
        self._field_decorations: dict[str, FieldDecoration] = {}
        self._choice_state: tuple[tuple[str, str], ...] = ()
        self._pending_section_refresh_paths: set[str] = set()
        self._section_refresh_scheduled = False
        self._editing_enabled = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        self._inner = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(4, 4, 4, 4)
        self._inner_layout.setSpacing(4)
        self._inner_layout.addStretch()
        scroll.setWidget(self._inner)

    def attach(self, draft: CfgDraft) -> None:
        """Render + reflect a service-owned ``CfgDraft``.

        Connects to the model's signals and builds the widget tree. The widget
        does NOT own the model — ``detach`` (and Qt destruction) never tears it
        down; only the ``CfgEditorService`` does.
        """
        self.detach()

        self._draft = draft
        self._field_decorations = {}
        self._choice_state = _choice_state_for_model(draft.root)
        draft.on_validity_changed.connect(self.validity_changed.emit)
        draft.on_change.connect(self._on_draft_changed)

        self._root_widget = SectionWidget(
            draft.root,
            top_level=True,
            field_label_max_width=self._field_label_max_width,
            decoration_for_path=self._resolve_decoration,
            text_input_enhancer=self._text_input_enhancer,
        )
        self._apply_editing_enabled()
        self._inner_layout.insertWidget(
            self._inner_layout.count() - 1, self._root_widget
        )

        self.validity_changed.emit(draft.is_valid())
        logger.debug("CfgFormWidget.attach: built reactive form over owned draft")

    def set_editing_enabled(self, enabled: bool) -> None:
        """Enable or disable editing without disabling the scroll container."""
        self._editing_enabled = bool(enabled)
        self._apply_editing_enabled()

    def detach(self) -> None:
        """Disconnect from the model + drop the widget tree, WITHOUT teardown.

        The model is service-owned and may outlive this widget (agent still
        editing it); only its signal bindings + the Qt widget tree go away.
        """
        if self._draft is not None:
            self._draft.on_change.disconnect(self._on_draft_changed)
            self._draft.on_validity_changed.disconnect(self.validity_changed.emit)
            self._draft = None  # NOTE: detach never closes a service-owned draft.

        if self._root_widget:
            self._root_widget.teardown()
            self._inner_layout.removeWidget(self._root_widget)
            self._root_widget.deleteLater()
            self._root_widget = None
        self._field_decorations = {}
        self._choice_state = ()
        self._pending_section_refresh_paths = set()
        self._section_refresh_scheduled = False

    def _apply_editing_enabled(self) -> None:
        root = self._root_widget
        if root is not None:
            root.setEnabled(self._editing_enabled)

    def set_decoration_provider(self, provider: FieldDecorationProvider | None) -> None:
        self._decoration_provider = provider
        draft = self._draft
        if draft is None or self._root_widget is None:
            return
        changed_paths = _changed_decoration_paths(
            self._field_decorations, self._decoration_state_for_model(draft.root)
        )
        self._queue_section_refresh(_parent_section_paths(changed_paths))

    def decoration_for_path(self, path: str) -> FieldDecoration:
        self._flush_pending_section_refresh()
        try:
            return self._field_decorations[path]
        except KeyError as exc:
            raise KeyError(f"Unknown cfg field path {path!r}") from exc

    def decoration_paths(self) -> tuple[str, ...]:
        self._flush_pending_section_refresh()
        return tuple(self._field_decorations)

    def read_values(self) -> CfgSectionValue:
        """Return a new CfgSectionValue from current model state."""
        if self._draft is None:
            raise RuntimeError("attach() must be called before read_values()")
        return self._draft.snapshot().value

    def read_schema(self) -> CfgSchema:
        """Snapshot the LiveModel draft as a fresh ``CfgSchema``.

        Tab callers should normally read ``State.cfg_schema`` instead (the
        committed truth, kept in sync by auto-commit). This method exists for
        local-draft hosts (inspect / writeback dialogs) that need to capture
        the unsaved draft at their own Apply boundary.
        """
        if self._draft is None:
            raise RuntimeError("attach() must be called before read_schema()")
        return self._draft.snapshot()

    def is_valid(self) -> bool:
        return self._draft.is_valid() if self._draft else True

    def first_invalid_reason(self) -> str | None:
        if self._draft is None:
            return None
        return self._find_first_invalid(self._draft.root, path="")

    def _emit_schema_changed(self, *_: object) -> None:
        if self._draft is None:
            return
        self.schema_changed.emit(self.read_schema())

    def _on_draft_changed(self, *_: object) -> None:
        if self._draft is None:
            return
        self.schema_changed.emit(self.read_schema())
        state = _choice_state_for_model(self._draft.root)
        if state == self._choice_state:
            return
        changed_selectors = _changed_choice_selector_paths(self._choice_state, state)
        self._choice_state = state
        self._queue_section_refresh(_parent_section_paths(changed_selectors))

    def _resolve_decoration(self, path: str, field: CfgField) -> FieldDecoration:
        decoration = self._decoration_for_field(path, field)
        self._field_decorations[path] = decoration
        return decoration

    def _decoration_for_field(self, path: str, field: CfgField) -> FieldDecoration:
        spec = cast(CfgNodeSpec, field.spec)
        value = cast(CfgNodeValue | None, field.get_value())
        provider = self._decoration_provider
        patch = None if provider is None else provider.decoration_for(path, spec, value)
        return default_decoration_for_spec(spec).merge(patch)

    def _decoration_state_for_model(
        self, model: SectionField
    ) -> dict[str, FieldDecoration]:
        state: dict[str, FieldDecoration] = {}
        self._collect_decoration_state(model, "", state)
        return state

    def _collect_decoration_state(
        self, field: CfgField, path: str, state: dict[str, FieldDecoration]
    ) -> None:
        if path:
            state[path] = self._decoration_for_field(path, field)
        if isinstance(field, SectionField):
            for key, child in field.fields.items():
                child_path = f"{path}.{key}" if path else key
                self._collect_decoration_state(child, child_path, state)
            return
        if isinstance(field, ReferenceField) and field.sub_field is not None:
            for key, child in field.sub_field.fields.items():
                child_path = f"{path}.{key}" if path else key
                self._collect_decoration_state(child, child_path, state)
            return
        if isinstance(field, SweepField):
            for edge, child in (
                ("start", field.start_field),
                ("stop", field.stop_field),
            ):
                child_path = f"{path}.{edge}" if path else edge
                state[child_path] = self._decoration_for_field(child_path, child)

    def _queue_section_refresh(self, section_paths: set[str]) -> None:
        if not section_paths:
            return
        self._pending_section_refresh_paths.update(section_paths)
        if self._section_refresh_scheduled:
            return
        self._section_refresh_scheduled = True
        QTimer.singleShot(0, self._flush_pending_section_refresh)

    def _flush_pending_section_refresh(self) -> None:
        if not self._pending_section_refresh_paths:
            self._section_refresh_scheduled = False
            return
        section_paths = set(self._pending_section_refresh_paths)
        self._pending_section_refresh_paths = set()
        self._section_refresh_scheduled = False
        self._refresh_section_paths(section_paths)

    def _refresh_section_paths(self, section_paths: set[str]) -> None:
        root = self._root_widget
        if root is None:
            return
        for path in _minimal_section_paths(section_paths):
            self._drop_decorations_under(path)
            if not root.refresh_section(path):
                # A stale or unsupported path should not leave the form half-updated.
                if self._draft is not None:
                    self.attach(self._draft)
                return

    def _drop_decorations_under(self, section_path: str) -> None:
        if not section_path:
            self._field_decorations = {}
            return
        prefix = f"{section_path}."
        self._field_decorations = {
            path: decoration
            for path, decoration in self._field_decorations.items()
            if not path.startswith(prefix)
        }

    def _find_first_invalid(self, field: CfgField, *, path: str) -> str | None:
        if isinstance(field, ScalarField):
            if field.is_valid():
                return None
            val = field.get_value()
            if isinstance(val, EvalValue) and val.error:
                return f"{path or field.spec.label}: {val.error}"
            return f"{path or field.spec.label}: invalid scalar value"

        if isinstance(field, SweepField):
            if field.start_field.is_valid() and field.stop_field.is_valid():
                return None
            start_reason = self._find_first_invalid(
                field.start_field,
                path=f"{path}.start" if path else "sweep.start",
            )
            if start_reason:
                return start_reason
            return self._find_first_invalid(
                field.stop_field,
                path=f"{path}.stop" if path else "sweep.stop",
            )

        if isinstance(field, CenteredSweepField):
            if field.center_field.is_valid():
                return None
            return self._find_first_invalid(
                field.center_field,
                path=f"{path}.center" if path else "sweep.center",
            )

        if isinstance(field, ReferenceField):
            if field.is_valid():
                return None
            key = field.get_chosen_key()
            label = path or field.spec.label
            if field.has_missing_library_ref():
                return f"{label}: missing library reference '{key}'"
            if field.sub_field is not None:
                nested = self._find_first_invalid(
                    field.sub_field, path=f"{label}.{key}"
                )
                if nested:
                    return nested
            return f"{label}: invalid module reference '{key}'"

        if isinstance(field, SectionField):
            for key, child in field.fields.items():
                child_path = f"{path}.{key}" if path else key
                reason = self._find_first_invalid(child, path=child_path)
                if reason:
                    return reason
            return None

        if not field.is_valid():
            return f"{path or type(field.spec).__name__}: invalid field"
        return None


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
    "CfgFormWidget",
    "FieldDecoration",
    "FieldDecorationPatch",
    "FieldDecorationProvider",
    "Tone",
    "default_decoration_for_spec",
]


def _choice_state_for_model(model: SectionField) -> tuple[tuple[str, str], ...]:
    state: list[tuple[str, str]] = []
    _collect_choice_state(model, "", state)
    return tuple(sorted(state))


def _collect_choice_state(
    field: SectionField, path: str, state: list[tuple[str, str]]
) -> None:
    spec = field.spec
    if isinstance(spec, ChoiceSectionSpec):
        for binding in spec.bindings:
            selector = field.fields.get(binding.selector_key)
            value = selector.get_value() if selector is not None else None
            choice = str(value.value) if isinstance(value, DirectValue) else ""
            selector_path = (
                f"{path}.{binding.selector_key}" if path else binding.selector_key
            )
            state.append((selector_path, choice))
    for key, child in field.fields.items():
        if isinstance(child, SectionField):
            child_path = f"{path}.{key}" if path else key
            _collect_choice_state(child, child_path, state)


def _changed_choice_selector_paths(
    old: tuple[tuple[str, str], ...], new: tuple[tuple[str, str], ...]
) -> set[str]:
    old_map = dict(old)
    new_map = dict(new)
    return {
        path
        for path in set(old_map) | set(new_map)
        if old_map.get(path) != new_map.get(path)
    }


def _changed_decoration_paths(
    old: dict[str, FieldDecoration], new: dict[str, FieldDecoration]
) -> set[str]:
    return {path for path in set(old) | set(new) if old.get(path) != new.get(path)}


def _parent_section_paths(paths: set[str]) -> set[str]:
    section_paths: set[str] = set()
    sweep_edges = {"center", "span", "start", "stop", "expts", "step"}
    for path in paths:
        section, sep, _leaf = path.rpartition(".")
        if sep and _leaf in sweep_edges:
            section, _sep, _sweep_leaf = section.rpartition(".")
        section_paths.add(section if sep else "")
    return section_paths


def _minimal_section_paths(paths: set[str]) -> tuple[str, ...]:
    ordered = sorted(paths, key=lambda p: (p.count("."), p))
    minimal: list[str] = []
    for path in ordered:
        if not path:
            return ("",)
        if any(path == parent or path.startswith(f"{parent}.") for parent in minimal):
            continue
        minimal.append(path)
    return tuple(minimal)
