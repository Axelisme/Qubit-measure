"""CfgFormWidget — renders a CfgSchema as an interactive reactive Qt form.

Uses LiveModel as the active data layer and delegates field rendering to
lib/zcu_tools/gui/ui/fields/.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from qtpy.QtCore import Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.adapter import EvalValue

from ..live_model import (
    LiveField,
    LiveModelEnv,
    ModuleRefLiveField,
    ScalarLiveField,
    SectionLiveField,
    SweepLiveField,
)
from .fields import SectionWidget

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import CfgSchema, CfgSectionValue
    from zcu_tools.gui.controller import Controller
    from zcu_tools.gui.event_bus import EventBus, GuiEvent

logger = logging.getLogger(__name__)


class CfgFormWidget(QWidget):
    """Container for the reactive experiment configuration form.

    Builds and owns a LiveModel (the runtime *draft* SSOT). Whether changes
    propagate to ``State.cfg_schema`` depends on who handles ``schema_changed``:

    - In the tab pane (``ExpTabWidget``), ``schema_changed`` is bound to
      ``Controller.update_tab_cfg`` — this is the auto-commit boundary.
    - In inspect / writeback dialogs, the host stores the draft locally and
      only commits on an explicit Apply.
    """

    validity_changed: Signal = Signal(bool)
    # Auto-commit signal (in tab mode) / draft-change notification (in dialog
    # mode). Payload is a freshly built CfgSchema snapshot of the LiveModel.
    schema_changed: Signal = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model: Optional[SectionLiveField] = None
        self._root_widget: Optional[SectionWidget] = None
        self._bus: Optional[EventBus] = None
        self._bus_subs: list[tuple[GuiEvent, Callable[..., None]]] = []

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

        # Unsubscribe from EventBus when Qt destroys this widget (e.g. via
        # deleteLater after tab close) so dead C++ callbacks are never invoked.
        self.destroyed.connect(self._unsubscribe_external_refresh)

    def populate(
        self,
        schema: CfgSchema,
        ctrl: Controller,
    ) -> None:
        """Build LiveModel and widget tree from schema."""
        self.clear()

        # 1. Create the environment and reactive data layer
        env = LiveModelEnv(ctrl=ctrl)
        self._model = SectionLiveField(schema.spec, env, initial_val=schema.value)
        self._model.on_validity_changed.connect(self.validity_changed.emit)
        self._model.on_change.connect(self._emit_schema_changed)
        self._subscribe_external_refresh(ctrl)

        # 2. Build the UI tree
        self._root_widget = SectionWidget(self._model, top_level=True)
        self._inner_layout.insertWidget(
            self._inner_layout.count() - 1, self._root_widget
        )

        # 3. Emit initial validity
        self.validity_changed.emit(self._model.is_valid())
        logger.debug("CfgFormWidget.populate: built reactive form")

    def clear(self) -> None:
        self._unsubscribe_external_refresh()
        if self._model:
            self._model.on_change.disconnect(self._emit_schema_changed)
            self._model.on_validity_changed.disconnect(self.validity_changed.emit)
            self._model.teardown()
            self._model = None

        if self._root_widget:
            self._inner_layout.removeWidget(self._root_widget)
            self._root_widget.deleteLater()
            self._root_widget = None

    def read_values(self) -> CfgSectionValue:
        """Return a new CfgSectionValue from current model state."""
        if self._model is None:
            raise RuntimeError("populate() must be called before read_values()")
        return self._model.get_value()

    def read_schema(self) -> CfgSchema:
        """Snapshot the LiveModel draft as a fresh ``CfgSchema``.

        Tab callers should normally read ``State.cfg_schema`` instead (the
        committed truth, kept in sync by auto-commit). This method exists for
        local-draft hosts (inspect / writeback dialogs) that need to capture
        the unsaved draft at their own Apply boundary.
        """
        from zcu_tools.gui.adapter import CfgSchema

        if self._model is None:
            raise RuntimeError("populate() must be called before read_schema()")
        return CfgSchema(spec=self._model.spec, value=self.read_values())

    def get_live_root(self) -> Optional[SectionLiveField]:
        """Return the live ``SectionLiveField`` root, or ``None`` if unpopulated.

        Exposed so the remote-control path resolver (Phase 81b) can mutate
        the draft tree directly — preserving WYSIWYG with the form widget,
        which auto-commits via ``schema_changed`` exactly as a UI edit would.
        """
        return self._model

    def is_valid(self) -> bool:
        return self._model.is_valid() if self._model else True

    def first_invalid_reason(self) -> Optional[str]:
        if self._model is None:
            return None
        return self._find_first_invalid(self._model, path="")

    def refresh_external(self, event: "GuiEvent") -> None:
        if self._model is None:
            return
        self._model.refresh_external(event)
        self.validity_changed.emit(self._model.is_valid())

    def _emit_schema_changed(self, *_: object) -> None:
        if self._model is None:
            return
        self.schema_changed.emit(self.read_schema())

    def _subscribe_external_refresh(self, ctrl: Controller) -> None:
        from zcu_tools.gui.event_bus import GuiEvent

        self._bus = ctrl.get_bus()
        for event in (
            GuiEvent.MD_CHANGED,
            GuiEvent.CONTEXT_SWITCHED,
            GuiEvent.ML_CHANGED,
            GuiEvent.DEVICE_CHANGED,
        ):
            cb = self._make_external_refresh_cb(event)
            self._bus.subscribe(event, cb)
            self._bus_subs.append((event, cb))

    def _unsubscribe_external_refresh(self) -> None:
        if self._bus is None:
            self._bus_subs.clear()
            return
        for event, cb in self._bus_subs:
            self._bus.unsubscribe(event, cb)
        self._bus_subs.clear()
        self._bus = None

    def _make_external_refresh_cb(self, event: "GuiEvent") -> "Callable[[Any], None]":
        def _callback(payload: Any) -> None:
            del payload
            self.refresh_external(event)

        return _callback

    def _find_first_invalid(self, field: LiveField, *, path: str) -> Optional[str]:
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
