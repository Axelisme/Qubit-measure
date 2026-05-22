"""CfgFormWidget — renders a CfgSchema as an interactive reactive Qt form.

REFACTORED (Phase 35):
- Uses LiveModel as the active data layer.
- Reactive fields handle their own UI synchronization.
- Decoupled widget implementation into lib/zcu_tools/gui/ui/fields/.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..live_model import SectionLiveField
from .fields import SectionWidget

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import CfgSchema, CfgSectionValue
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.meta_tool import MetaDict
    from zcu_tools.meta_tool.library import ModuleLibrary

logger = logging.getLogger(__name__)


class CfgFormWidget(QWidget):
    """Container for the reactive experiment configuration form."""

    validity_changed: Signal = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model: Optional[SectionLiveField] = None
        self._root_widget: Optional[SectionWidget] = None

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

    def populate(
        self,
        schema: CfgSchema,
        bus: EventBus,
        ml: Optional[ModuleLibrary] = None,
        md: Optional[MetaDict] = None,
    ) -> None:
        """Build LiveModel and widget tree from schema."""
        self._clear_inner()

        # 1. Create the reactive data layer
        self._model = SectionLiveField(
            schema.spec, bus, ml=ml, md=md, initial_val=schema.value
        )
        self._model.on_validity_changed.connect(self.validity_changed.emit)

        # 2. Build the UI tree
        self._root_widget = SectionWidget(self._model, top_level=True)
        self._inner_layout.insertWidget(self._inner_layout.count() - 1, self._root_widget)
        
        # 3. Emit initial validity
        self.validity_changed.emit(self._model.is_valid())
        logger.debug("CfgFormWidget.populate: built reactive form")

    def _clear_inner(self) -> None:
        if self._model:
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

    def to_dict(self) -> dict[str, Any]:
        """Convenience: return raw dict for experiment runner."""
        return self._model.to_dict() if self._model else {}
