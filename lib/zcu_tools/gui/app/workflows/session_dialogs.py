"""Retained shared Setup/SoC dialogs for the workflow host."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QWidget

from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
from zcu_tools.gui.session.ui.soc_dialog import SocDialog

if TYPE_CHECKING:
    from zcu_tools.gui.session.services.build import SessionServices


class WorkflowSessionDialogs:
    """Own the workflow host's retained Setup and SoC dialog pair.

    setup_dialog manages entry/point inputs; soc_dialog manages connection and
    configuration. Both are parented to the Window and retained after Close.
    All methods run on the GUI owner thread. Read-only mode locks submissions,
    while browsing and Close remain available. Closing does not disconnect
    hardware or replace workflow form drafts.
    """

    def __init__(
        self,
        services: SessionServices,
        on_setup_finished: Callable[[], None],
        parent: QWidget,
    ) -> None:
        """Build both dialogs from the host's services, with parent owning Qt lifetime.

        on_setup_finished refreshes the host's session chips whenever Setup
        finishes. Construction and showing do not open an entry or connect hardware.
        """
        self.setup_dialog = SetupDialog(services.setup_control, parent)
        self.setup_dialog.finished.connect(on_setup_finished)
        self.soc_dialog = SocDialog(services.soc_control, parent)

    def open_setup(self) -> None:
        """Show the same retained shared Setup non-modally on the GUI owner.

        Reopening refreshes the dialog's cached inputs, not form captures. No
        entry is opened or hardware connected by showing. Closing Setup refreshes
        the session chips without replacing form drafts.
        """
        self.setup_dialog.show()
        self.setup_dialog.raise_()
        self.setup_dialog.activateWindow()

    def open_soc(self) -> None:
        """Show the retained non-modal SoC connection/configuration dialog.

        Reopening prefills cached connection inputs. The host keeps submissions
        read-only while a run holds the session; configuration browsing and
        Close remain available.
        """
        self.soc_dialog.show()
        self.soc_dialog.raise_()
        self.soc_dialog.activateWindow()

    def set_read_only(self, *, read_only: bool) -> None:
        """Lock or unlock submissions in both retained dialogs, preserving browsing."""
        self.setup_dialog.set_read_only(read_only=read_only)
        self.soc_dialog.set_read_only(read_only=read_only)

    def close(self) -> None:
        """Close both dialogs after host shutdown succeeds, retaining their instances."""
        self.setup_dialog.close()
        self.soc_dialog.close()
