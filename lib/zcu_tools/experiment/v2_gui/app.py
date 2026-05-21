"""App entry point — assembles all components and launches the GUI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from zcu_tools.gui.controller import Controller

from zcu_tools.gui.adapter import ExpContext
from zcu_tools.gui.controller import Controller
from zcu_tools.gui.device_manager import DeviceManager
from zcu_tools.gui.io_manager import IOManager
from zcu_tools.gui.registry import Registry
from zcu_tools.gui.runner import Runner
from zcu_tools.gui.state import State
from zcu_tools.gui.ui.main_window import MainWindow

from .registry import register_all


def _make_empty_ctx() -> ExpContext:
    """Minimal startup context: real empty MetaDict/ModuleLibrary, no file sync."""
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    return ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
    )


def run_app() -> None:
    """Build and launch the GUI. Blocks until the window is closed."""
    app = QApplication.instance() or QApplication(sys.argv)

    # --- wire up components ---
    registry = Registry()
    register_all(registry)

    state = State(_make_empty_ctx())
    runner = Runner()
    io_manager = IOManager()
    device_manager = DeviceManager()

    # MainWindow needs a Controller reference, but Controller needs a View.
    # Use a two-step init: create window stub, then build controller.
    # We pass a forward-reference sentinel and patch after construction.
    window = _build_window(state, runner, registry, io_manager, device_manager)
    window.show()

    # Show startup dialog to let user set chip/qub names and derive paths.
    # Non-blocking: user can close it and still use the app (with empty context).
    _show_startup_dialog(window._ctrl, parent=window)  # type: ignore[attr-defined]

    sys.exit(app.exec())


def _show_startup_dialog(ctrl: "Controller", parent: Any) -> None:
    from zcu_tools.gui.ui.setup_dialog import SetupDialog

    dlg = SetupDialog(ctrl, parent=parent, startup_mode=True)
    dlg.exec()


def _build_window(
    state: State,
    runner: Runner,
    registry: Registry,
    io_manager: IOManager,
    device_manager: DeviceManager,
) -> MainWindow:
    """Create Controller + MainWindow in the correct order."""

    # Temporary placeholder View that satisfies ViewProtocol before the real
    # window is ready — used only during Controller.__init__ signal wiring.
    class _PlaceholderView:
        def refresh_tab(self, tab_id: str) -> None:
            _ = tab_id

        def refresh_run_state(self, is_running: bool) -> None:
            _ = is_running

        def refresh_context_panel(self) -> None: ...

        def refresh_config_panels(self) -> None: ...

        def refresh_predictor_panel(self) -> None: ...

        def make_pbar_factory(self, tab_id: str) -> None:
            _ = tab_id
            return None  # type: ignore[return-value]

        def make_live_container(self, tab_id: str) -> None:
            _ = tab_id
            return None  # type: ignore[return-value]

        def show_status_message(self, message: str) -> None:
            _ = message

    ctrl = Controller(
        state=state,
        runner=runner,
        registry=registry,
        io_manager=io_manager,
        device_manager=device_manager,
        view=_PlaceholderView(),  # type: ignore[arg-type]
    )

    window = MainWindow(ctrl)
    # Replace placeholder with the real window
    ctrl._view = window  # type: ignore[attr-defined]
    return window
