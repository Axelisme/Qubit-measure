"""GUI composition root — assembles all components for the shared runtime.

This module depends only on ``zcu_tools.gui.app.measure``; it does not know which concrete
experiments exist. The entry script wires a populated ``Registry`` /
``TemplateCatalog`` (built from caller-owned definitions) and passes them in — so the
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
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.app.measure.catalog import ExperimentCatalogLoader
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.app.measure.registry import Registry
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.session.services.io_manager import IOManager

RegistryFactory = Callable[
    [], tuple["Registry", "TemplateCatalog", "ExperimentCatalogLoader"]
]


def _make_empty_ctx() -> SessionEnv:
    """Minimal initial context: real empty MetaDict/ModuleLibrary, no file sync."""
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.resources.context import MetaDict, ModuleLibrary

    return SessionEnv(
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
        default_control_port=8765,
    )

    def __init__(
        self,
        registry_factory: RegistryFactory,
        *,
        clean: bool = False,
        project_root: str | None = None,
    ) -> None:
        from zcu_tools.gui.app.measure.ui.error_handler import show_error_dialog
        from zcu_tools.gui.app.measure.utils.error_handler import (
            install_global_exception_hook,
        )

        install_global_exception_hook(show_error_dialog)
        self._registry, self._template_catalog, self._catalog_loader = (
            registry_factory()
        )
        self._clean = clean
        self._project_root = project_root

    def assemble(self, control: ControlOptions | None) -> GuiAssembly:
        from zcu_tools.gui.app.measure.state import State
        from zcu_tools.gui.session.services.io_manager import IOManager

        state = State(_make_empty_ctx())
        io_manager = IOManager()
        ctrl, window = _build_window(
            state,
            self._registry,
            self._template_catalog,
            io_manager,
            self._project_root,
            catalog_loader=self._catalog_loader,
        )

        adapter = None
        if control is not None:
            from zcu_tools.gui.app.measure.remote import RemoteControlAdapter

            adapter = RemoteControlAdapter(
                controller=ctrl,
                opts=control,
                owner_scheduler=ctrl.owner_scheduler,
                render_view=window,
            )
            # MainWindow._perform_close stops this before accepting close. The
            # runtime also connects adapter.stop to aboutToQuit as an idempotent
            # safety net for non-window shutdown paths.
            window.remote_control_service = adapter  # type: ignore[attr-defined]

        return GuiAssembly(controller=ctrl, window=window, control_adapter=adapter)

    def before_show(self, assembly: GuiAssembly) -> None:
        from zcu_tools.gui.app.measure.services import create_persistence_caretaker

        ctrl = assembly.controller
        assert _is_controller(ctrl)
        caretaker = create_persistence_caretaker(ctrl)
        ctrl.attach_caretaker(caretaker)
        ctrl.restore_all(load=not self._clean)

    def after_show(self, assembly: GuiAssembly) -> None:
        from zcu_tools.gui.app.measure.remote.dialogs import DialogName

        parent = assembly.window
        assert _is_main_window(parent)
        # The same Setup a toolbar click opens: non-modal so the Qt event loop
        # (and the remote control socket) keeps pumping while it is visible.
        parent.open_dialog(DialogName.SETUP)


def _is_controller(value: object) -> TypeGuard[Controller]:
    from zcu_tools.gui.app.measure.controller import Controller

    return isinstance(value, Controller)


def _is_main_window(value: object) -> TypeGuard[MainWindow]:
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    return isinstance(value, MainWindow)


def _build_window(
    state: State,
    registry: Registry,
    template_catalog: TemplateCatalog,
    io_manager: IOManager,
    project_root: str | None = None,
    *,
    catalog_loader: ExperimentCatalogLoader | None = None,
) -> tuple[Controller, MainWindow]:
    """Create Controller + MainWindow in the correct order."""

    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.event_bus import BaseEventBus

    bus = BaseEventBus()

    ctrl = Controller(
        state=state,
        registry=registry,
        template_catalog=template_catalog,
        io_manager=io_manager,
        view=None,
        bus=bus,
        project_root=project_root,
        catalog_loader=catalog_loader,
    )

    window = MainWindow(ctrl)
    ctrl.add_view(window)
    return ctrl, window
