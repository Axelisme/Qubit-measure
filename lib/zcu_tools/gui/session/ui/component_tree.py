"""Component tree frontend for the session's parameter owner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QWidget

if TYPE_CHECKING:
    from zcu_tools.format_version import YamlMap, YamlValue
    from zcu_tools.gui.session.context_control import ContextControlPort


class ComponentTreeWidget(QWidget):
    """Inspect and edit registered components through ContextControlPort.

    The widget owns presentation only. Tree rows show component names, kind,
    fields and working-unit values. Missing required fields are red in the
    creation form. References use ordinary text, not graph-aware pickers.
    A setup/current selector controls editing destination. All writes use the
    service, and errors remain visible without mutating the rejected document.
    """

    def __init__(
        self, control: ContextControlPort, parent: QWidget | None = None
    ) -> None:
        """Bind the owner-thread control and render its current document.

        EMPTY displays no editable rows. Creation chooses a registered kind and
        collects required fields before Add. The current point view permits
        selecting a component as focus; setup mode does not change focus.
        """
        super().__init__(parent)
        raise NotImplementedError

    def refresh(self) -> None:
        """Render the cached selected document; never implicitly refresh files."""
        raise NotImplementedError

    def add_component(
        self, name: str, kind: str, fields: YamlMap, *, setup: bool = False
    ) -> bool:
        """Submit Add and refresh on success, returning True.

        Recoverable validation/resource failures display red text in the
        component_error QLabel and return False. Input fields stay available
        for correction; no default or empty value hides the failure.
        """
        raise NotImplementedError

    def edit_parameter(
        self, path: str, value: YamlValue, *, setup: bool = False
    ) -> bool:
        """Edit a concrete field through its owner; report errors as for Add."""
        raise NotImplementedError

    def set_focus(self, name: str) -> None:
        """Select a current component through the owner; errors propagate."""
        raise NotImplementedError
