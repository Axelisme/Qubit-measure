"""CfgFormWidget — renders a CfgSchema as an interactive reactive Qt form.

Uses LiveModel as the active data layer and delegates field rendering to
lib/zcu_tools/gui/ui/fields/.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.main.adapter import EvalValue

from ..live_model import (
    LiveField,
    ModuleRefLiveField,
    ScalarLiveField,
    SectionLiveField,
    SweepLiveField,
)
from .fields import SectionWidget

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import CfgSchema, CfgSectionValue

logger = logging.getLogger(__name__)


class CfgFormWidget(QWidget):
    """A pluggable viewer over a service-owned cfg LiveModel (ADR-0008).

    The widget does *not* own a LiveModel — it ``attach``es to one that the
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model: SectionLiveField | None = None
        self._root_widget: SectionWidget | None = None

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

    def attach(self, model: SectionLiveField) -> None:
        """Render + reflect a service-owned ``SectionLiveField``.

        Connects to the model's signals and builds the widget tree. The widget
        does NOT own the model — ``detach`` (and Qt destruction) never tears it
        down; only the ``CfgEditorService`` does.
        """
        self.detach()

        self._model = model
        model.on_validity_changed.connect(self.validity_changed.emit)
        model.on_change.connect(self._emit_schema_changed)

        self._root_widget = SectionWidget(model, top_level=True)
        self._inner_layout.insertWidget(
            self._inner_layout.count() - 1, self._root_widget
        )

        self.validity_changed.emit(model.is_valid())
        logger.debug("CfgFormWidget.attach: built reactive form over owned model")

    def detach(self) -> None:
        """Disconnect from the model + drop the widget tree, WITHOUT teardown.

        The model is service-owned and may outlive this widget (agent still
        editing it); only its signal bindings + the Qt widget tree go away.
        """
        if self._model is not None:
            self._model.on_change.disconnect(self._emit_schema_changed)
            self._model.on_validity_changed.disconnect(self.validity_changed.emit)
            self._model = None  # NOTE: never self._model.teardown()

        if self._root_widget:
            self._root_widget.teardown()
            self._inner_layout.removeWidget(self._root_widget)
            self._root_widget.deleteLater()
            self._root_widget = None

    def read_values(self) -> CfgSectionValue:
        """Return a new CfgSectionValue from current model state."""
        if self._model is None:
            raise RuntimeError("attach() must be called before read_values()")
        return self._model.get_value()

    def read_schema(self) -> CfgSchema:
        """Snapshot the LiveModel draft as a fresh ``CfgSchema``.

        Tab callers should normally read ``State.cfg_schema`` instead (the
        committed truth, kept in sync by auto-commit). This method exists for
        local-draft hosts (inspect / writeback dialogs) that need to capture
        the unsaved draft at their own Apply boundary.
        """
        from zcu_tools.gui.app.main.adapter import CfgSchema

        if self._model is None:
            raise RuntimeError("attach() must be called before read_schema()")
        return CfgSchema(spec=self._model.spec, value=self.read_values())

    def get_live_root(self) -> SectionLiveField | None:
        """Return the live ``SectionLiveField`` root, or ``None`` if unpopulated.

        Exposed so the remote-control path resolver (Phase 81b) can mutate
        the draft tree directly — preserving WYSIWYG with the form widget,
        which auto-commits via ``schema_changed`` exactly as a UI edit would.
        """
        return self._model

    def is_valid(self) -> bool:
        return self._model.is_valid() if self._model else True

    def first_invalid_reason(self) -> str | None:
        if self._model is None:
            return None
        return self._find_first_invalid(self._model, path="")

    def _emit_schema_changed(self, *_: object) -> None:
        if self._model is None:
            return
        self.schema_changed.emit(self.read_schema())

    def _find_first_invalid(self, field: LiveField, *, path: str) -> str | None:
        if isinstance(field, ScalarLiveField):
            if field.is_valid():
                return None
            val = field.get_value()
            if isinstance(val, EvalValue) and val.error:
                return f"{path or field.spec.label}: {val.error}"
            return f"{path or field.spec.label}: invalid scalar value"

        if isinstance(field, SweepLiveField):
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

        if isinstance(field, ModuleRefLiveField):
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

        if isinstance(field, SectionLiveField):
            for key, child in field.fields.items():
                child_path = f"{path}.{key}" if path else key
                reason = self._find_first_invalid(child, path=child_path)
                if reason:
                    return reason
            return None

        if not field.is_valid():
            return f"{path or type(field.spec).__name__}: invalid field"
        return None
