"""GUI composition root — assembles all components and launches the app.

This module depends only on ``zcu_tools.gui.app.main``; it does not know which concrete
experiments exist. The entry script wires a populated ``Registry`` /
``RoleCatalog`` (built from ``experiment.v2_gui``) and passes them in — so the
GUI framework never imports the experiment-adapter layer.

The matplotlib backend must already be configured before ``run_app`` runs (the
entry script calls ``configure_matplotlib_backend`` from
``zcu_tools.gui.plotting.setup`` before importing this module); this module only
assembles and launches, it does not configure the
backend itself.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.registry import Registry
    from zcu_tools.gui.app.main.role_catalog import RoleCatalog
    from zcu_tools.gui.app.main.services.remote import ControlOptions
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.app.main.ui.main_window import MainWindow
    from zcu_tools.gui.session.services.io_manager import IOManager


def _make_empty_ctx() -> ExpContext:
    """Minimal startup context: real empty MetaDict/ModuleLibrary, no file sync."""
    from zcu_tools.gui.app.main.adapter import ExpContext
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    return ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
    )


def run_app(
    registry: Registry,
    role_catalog: RoleCatalog,
    control_opts: ControlOptions | None = None,
    clean: bool = False,
    project_root: str | None = None,
) -> None:
    """Build and launch the GUI. Blocks until the window is closed.

    ``registry`` and ``role_catalog`` are already populated by the entry script
    (the GUI framework does not know which experiments exist — the script wires
    them from ``experiment.v2_gui``).

    ``control_opts`` (if provided) starts a ``RemoteControlAdapter`` after the
    window is constructed; the adapter is stopped from ``MainWindow.closeEvent``.

    ``clean=True`` starts without restoring the previous persisted session
    (the on-disk ``gui_state_v1.json`` is left untouched at startup; a normal
    close still flushes over it).

    ``project_root`` is the base dir the default result/database paths anchor
    under; the entry script passes the repo root so a .bat launcher that cd's
    into script/ does not scope defaults under script/. None falls back to cwd.
    """
    from zcu_tools.gui.app.main.utils.error_handler import install_global_exception_hook

    install_global_exception_hook()

    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.session.services.io_manager import IOManager

    app = QApplication.instance() or QApplication(sys.argv)

    # --- wire up components ---
    state = State(_make_empty_ctx())
    io_manager = IOManager()

    ctrl, window = _build_window(
        state, registry, role_catalog, io_manager, project_root
    )

    # App-level PersistenceCaretaker (Memento Caretaker): owns the single
    # gui_state_v1.json file. The Controller is the Originator; restore at
    # startup, flush at close (MainWindow._perform_close → ctrl.persist_all).
    from zcu_tools.gui.app.main.services import PersistenceCaretaker

    caretaker = PersistenceCaretaker(ctrl)
    ctrl.attach_caretaker(caretaker)
    ctrl.restore_all(load=not clean)
    window.show()

    if control_opts is not None:
        from zcu_tools.gui.app.main.services.remote import RemoteControlAdapter

        adapter = RemoteControlAdapter(
            controller=ctrl, opts=control_opts, render_view=window
        )
        adapter.start()
        # Stash on the window so MainWindow.closeEvent can stop it; keep a
        # strong ref so the GC does not retire the daemon thread early.
        window.remote_control_service = adapter  # type: ignore[attr-defined]

    # Show startup dialog to let user set chip/qub names and derive paths.
    # Non-blocking: user can close it and still use the app (with empty context).
    _show_startup_dialog(ctrl, parent=window)

    sys.exit(app.exec())


def _show_startup_dialog(ctrl: Controller, parent: MainWindow) -> None:
    """Show the bootstrap startup dialog non-modally.

    Non-modal is required so the Qt event loop keeps pumping while the
    dialog is visible — this is what lets ``RemoteControlAdapter`` accept
    further RPCs (e.g. ``dialog.close STARTUP``) while a remote agent is
    driving onboarding. The dialog registers in ``window._open_dialogs``
    so it shows up in ``dialog.list_open`` queries.
    """
    from qtpy.QtCore import Qt  # type: ignore[attr-defined]

    from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
    from zcu_tools.gui.session.ui.setup_dialog import SetupDialog

    dlg = SetupDialog(ctrl, parent=parent, startup_mode=True)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    parent.register_dialog(DialogName.STARTUP, dlg)
    dlg.open()


def _build_window(
    state: State,
    registry: Registry,
    role_catalog: RoleCatalog,
    io_manager: IOManager,
    project_root: str | None = None,
) -> tuple[Controller, MainWindow]:
    """Create Controller + MainWindow in the correct order."""

    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.ui.main_window import MainWindow
    from zcu_tools.gui.event_bus import BaseEventBus

    bus = BaseEventBus()

    ctrl = Controller(
        state=state,
        registry=registry,
        role_catalog=role_catalog,
        io_manager=io_manager,
        view=None,
        bus=bus,
        project_root=project_root,
    )

    window = MainWindow(ctrl)
    ctrl.add_view(window)
    return ctrl, window
