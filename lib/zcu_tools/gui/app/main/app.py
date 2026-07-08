"""GUI composition root — assembles all components for the shared runtime.

This module depends only on ``zcu_tools.gui.app.main``; it does not know which concrete
experiments exist. The entry script wires a populated ``Registry`` /
``RoleCatalog`` (built from ``experiment.v2_gui``) and passes them in — so the
GUI framework never imports the experiment-adapter layer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, TypeGuard

from zcu_tools.gui.remote.rpc_endpoint import ControlOptions
from zcu_tools.gui.runtime import (
    GuiAssembly,
    GuiRuntimeBehavior,
    GuiRuntimeSpec,
    PlotPolicy,
    run_gui_runtime,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.registry import Registry
    from zcu_tools.gui.app.main.role_catalog import RoleCatalog
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.app.main.ui.main_window import MainWindow
    from zcu_tools.gui.session.services.io_manager import IOManager

RegistryFactory = Callable[[], tuple["Registry", "RoleCatalog"]]


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


class MeasureGuiBehavior(GuiRuntimeBehavior):
    """measure-gui app wiring behind the shared GUI runtime."""

    spec: ClassVar[GuiRuntimeSpec] = GuiRuntimeSpec(
        app_name="measure",
        app_slug="measure",
        plot_policy=PlotPolicy.EMBEDDED_BACKEND,
        default_control_port=8765,
        logging_extra_namespaces=("zcu_tools.experiment.v2_gui",),
    )

    def __init__(
        self,
        registry_factory: RegistryFactory,
        *,
        clean: bool = False,
        project_root: str | None = None,
    ) -> None:
        from zcu_tools.gui.app.main.utils.error_handler import (
            install_global_exception_hook,
        )

        install_global_exception_hook()
        self._registry, self._role_catalog = registry_factory()
        self._clean = clean
        self._project_root = project_root

    def assemble(self, control: ControlOptions | None) -> GuiAssembly:
        from zcu_tools.gui.app.main.state import State
        from zcu_tools.gui.session.services.io_manager import IOManager

        state = State(_make_empty_ctx())
        io_manager = IOManager()
        ctrl, window = _build_window(
            state,
            self._registry,
            self._role_catalog,
            io_manager,
            self._project_root,
        )

        adapter = None
        if control is not None:
            from zcu_tools.gui.app.main.services.remote import RemoteControlAdapter

            adapter = RemoteControlAdapter(
                controller=ctrl,
                opts=control,
                render_view=window,
            )
            # MainWindow._perform_close stops this before accepting close. The
            # runtime also connects adapter.stop to aboutToQuit as an idempotent
            # safety net for non-window shutdown paths.
            window.remote_control_service = adapter  # type: ignore[attr-defined]

        return GuiAssembly(controller=ctrl, window=window, control_adapter=adapter)

    def before_show(self, assembly: GuiAssembly) -> None:
        from zcu_tools.gui.app.main.services import PersistenceCaretaker

        ctrl = assembly.controller
        assert _is_controller(ctrl)
        caretaker = PersistenceCaretaker(ctrl)
        ctrl.attach_caretaker(caretaker)
        ctrl.restore_all(load=not self._clean)

    def after_show(self, assembly: GuiAssembly) -> None:
        ctrl = assembly.controller
        assert _is_controller(ctrl)
        parent = assembly.window
        assert _is_main_window(parent)
        _show_startup_dialog(ctrl, parent=parent)


def run_app(
    registry: Registry,
    role_catalog: RoleCatalog,
    control_opts: ControlOptions | None = None,
    clean: bool = False,
    project_root: str | None = None,
) -> int:
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
    return run_gui_runtime(
        MeasureGuiBehavior(
            lambda: (registry, role_catalog),
            clean=clean,
            project_root=project_root,
        ),
        control_opts,
    )


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

    dlg = SetupDialog(ctrl.setup_control, parent=parent, startup_mode=True)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    parent.register_dialog(DialogName.STARTUP, dlg)
    dlg.open()


def _is_controller(value: object) -> TypeGuard[Controller]:
    from zcu_tools.gui.app.main.controller import Controller

    return isinstance(value, Controller)


def _is_main_window(value: object) -> TypeGuard[MainWindow]:
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    return isinstance(value, MainWindow)


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
