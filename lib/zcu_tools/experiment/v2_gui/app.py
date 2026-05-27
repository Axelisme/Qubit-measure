"""App entry point — assembles all components and launches the GUI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext
    from zcu_tools.gui.controller import Controller
    from zcu_tools.gui.io_manager import IOManager
    from zcu_tools.gui.registry import Registry
    from zcu_tools.gui.runner import Runner
    from zcu_tools.gui.services.remote import ControlOptions
    from zcu_tools.gui.state import State
    from zcu_tools.gui.ui.main_window import MainWindow

from zcu_tools.gui.mpl_backend_setup import configure_gui_matplotlib_backend


def _make_empty_ctx() -> "ExpContext":
    """Minimal startup context: real empty MetaDict/ModuleLibrary, no file sync."""
    from zcu_tools.gui.adapter import ExpContext
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    return ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
    )


def run_app(control_opts: Optional["ControlOptions"] = None) -> None:
    """Build and launch the GUI. Blocks until the window is closed.

    ``control_opts`` (if provided) starts a ``RemoteControlService`` after the
    window is constructed; the service is stopped from ``MainWindow.closeEvent``.
    """
    from zcu_tools.gui.utils.error_handler import install_global_exception_hook

    install_global_exception_hook()

    configure_gui_matplotlib_backend()

    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.gui.io_manager import IOManager
    from zcu_tools.gui.registry import Registry
    from zcu_tools.gui.runner import Runner
    from zcu_tools.gui.state import State

    from .registry import register_all

    app = QApplication.instance() or QApplication(sys.argv)

    # --- wire up components ---
    registry = Registry()
    register_all(registry)

    state = State(_make_empty_ctx())
    runner = Runner()
    io_manager = IOManager()

    ctrl, window = _build_window(state, runner, registry, io_manager)
    ctrl.restore_tabs_from_session()
    ctrl.restore_startup_settings()
    window.show()

    if control_opts is not None:
        from zcu_tools.gui.services.remote import RemoteControlService

        service = RemoteControlService(controller=ctrl, opts=control_opts)
        service.start()
        # Stash on the window so MainWindow.closeEvent can stop it; keep a
        # strong ref so the GC does not retire the daemon thread early.
        window.remote_control_service = service  # type: ignore[attr-defined]

    # Show startup dialog to let user set chip/qub names and derive paths.
    # Non-blocking: user can close it and still use the app (with empty context).
    _show_startup_dialog(ctrl, parent=window)

    sys.exit(app.exec())


def _show_startup_dialog(ctrl: "Controller", parent: "MainWindow") -> None:
    from zcu_tools.gui.ui.setup_dialog import SetupDialog

    dlg = SetupDialog(ctrl, parent=parent, startup_mode=True)
    dlg.exec()


def _build_window(
    state: "State",
    runner: "Runner",
    registry: "Registry",
    io_manager: "IOManager",
) -> tuple["Controller", "MainWindow"]:
    """Create Controller + MainWindow in the correct order."""

    from zcu_tools.gui.controller import Controller
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.ui.main_window import MainWindow

    bus = EventBus()

    ctrl = Controller(
        state=state,
        runner=runner,
        registry=registry,
        io_manager=io_manager,
        view=None,
        bus=bus,
    )

    window = MainWindow(ctrl)
    ctrl.set_view(window)
    return ctrl, window
