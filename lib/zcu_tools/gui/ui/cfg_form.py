"""CfgFormWidget — renders a CfgSchema as an interactive reactive Qt form.

REFACTORED (Phase 35/36):
- Uses LiveModel as the active data layer.
- Reactive fields handle their own UI synchronization via LiveModelEnv.
- Decoupled widget implementation into lib/zcu_tools/gui/ui/fields/.
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

from ..live_model import LiveModelEnv, SectionLiveField
from .fields import SectionWidget

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import CfgSchema, CfgSectionValue
    from zcu_tools.gui.controller import Controller
    from zcu_tools.gui.event_bus import EventBus, GuiEvent

logger = logging.getLogger(__name__)


class CfgFormWidget(QWidget):
    """Container for the reactive experiment configuration form."""

    validity_changed: Signal = Signal(bool)
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
        self._clear_inner()

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

    def _clear_inner(self) -> None:
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
        """Return a new CfgSchema combining the stored spec with current model state."""
        from zcu_tools.gui.adapter import CfgSchema

        if self._model is None:
            raise RuntimeError("populate() must be called before read_schema()")
        return CfgSchema(spec=self._model.spec, value=self.read_values())

    def is_valid(self) -> bool:
        return self._model.is_valid() if self._model else True

    def _emit_schema_changed(self, *_: object) -> None:
        if self._model is None:
            return
        self.schema_changed.emit(self.read_schema())

    def _subscribe_external_refresh(self, ctrl: Controller) -> None:
        from zcu_tools.gui.event_bus import GuiEvent

        self._bus = ctrl.get_bus()
        for event in (
            GuiEvent.MD_CHANGED,
            GuiEvent.CONTEXT_CHANGED,
            GuiEvent.INSPECT_CHANGED,
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
        import weakref

        weak_self = weakref.ref(self)

        def _callback(payload: Any) -> None:
            del payload
            self_ref = weak_self()
            if self_ref is None:
                return

            try:
                # Test if the C++ QWidget object is still alive
                _ = self_ref.parent()
            except RuntimeError:
                # C++ object has been deleted, skip callback execution
                return

            if self_ref._model is None:
                return
            self_ref._model.refresh_external(event)
            self_ref.validity_changed.emit(self_ref._model.is_valid())

        return _callback

    def to_dict(self) -> dict[str, Any]:
        """Convenience: return raw dict for experiment runner."""
        return self._model.to_dict() if self._model else {}
