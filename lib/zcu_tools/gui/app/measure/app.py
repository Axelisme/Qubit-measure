"""GUI composition root — assembles all components for the shared runtime.

This module depends only on ``zcu_tools.gui.app.measure``; it does not know which concrete
experiments exist. The entry script wires a populated ``Registry`` with its injected templates
(built from caller-owned definitions) and passes it in — so the
GUI framework never imports the experiment-adapter layer.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeGuard

from zcu_tools.experiment.simulation import SimulationEnvironment, load_environment
from zcu_tools.gui.remote.rpc_endpoint import ControlOptions
from zcu_tools.gui.runtime import (
    GuiAssembly,
    GuiRuntimeBehavior,
    GuiRuntimeSpec,
)
from zcu_tools.gui.session.alias_presets import AliasPresets, validate_alias_presets

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.app.measure.catalog import ExperimentCatalogLoader
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.app.measure.registry import Registry
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.session.services.predictor import ParamsFor

RegistryFactory = Callable[[], tuple["Registry", "ExperimentCatalogLoader"]]


def _make_empty_ctx() -> SessionEnv:
    """EMPTY starts without parameter sources, hardware or file I/O."""
    from zcu_tools.gui.app.measure.adapter import SessionEnv

    return SessionEnv(parameters=None, soc=None, soccfg=None)


class MeasureGuiBehavior(GuiRuntimeBehavior):
    """measure-gui wiring; params_for optionally injects user-owned component prefill.

    registry_factory supplies catalogs after runtime setup. clean skips preference
    restore; project_root anchors entry roots. Predictor queries never depend on
    dialog state. Assembly/source failures propagate without hidden retries.
    """

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
        params_for: ParamsFor | None = None,
        alias_presets: AliasPresets | None = None,
        simulate: str | Path | None = None,
    ) -> None:
        """Build caller catalogs and retain complete alias data for assembly.

        ``registry_factory`` returns the registry (including templates) and reload loader
        after runtime setup. ``clean`` skips restoring persisted preferences;
        ``project_root`` anchors default paths, or uses the working directory.
        ``alias_presets`` maps preset names to ordered authored rows. Omitted
        or None means no catalog and an empty initial table. An explicit mapping
        must contain "default" (its rows may be empty), or raise ValueError.
        Invalid table syntax, duplicates or forward links raise InvalidInputError.
        The public ``alias_presets`` field retains the complete catalog for dialog
        composition, not a file location. Assembly uses default rows for the
        session's one-time unique-focus initialization. Saved entry tables take
        precedence, including empty tables. No device or cache binding occurs.
        ``simulate`` selects a trusted Python builder at assembly, once per process.
        Its successful return transfers ownership of the exact manager to this
        composition; startup failure propagates and triggers manager cleanup.
        """
        from zcu_tools.gui.app.measure.ui.error_handler import show_error_dialog
        from zcu_tools.gui.app.measure.utils.error_handler import (
            install_global_exception_hook,
        )

        validate_alias_presets(alias_presets)
        install_global_exception_hook(show_error_dialog)
        self._registry, self._catalog_loader = registry_factory()
        self._clean = clean
        self._project_root = project_root
        self._params_for = params_for
        self.alias_presets: AliasPresets | None = alias_presets
        self._simulate = simulate
        self._environment: SimulationEnvironment | None = None
        self._assembled = False
        self._environment_close_attempted = False
        self._controller: Controller | None = None

    def assemble(self, control: ControlOptions | None) -> GuiAssembly:
        if self._assembled:
            raise RuntimeError("measure behavior can only assemble once")
        self._assembled = True
        if self._simulate is not None:
            self._environment = load_environment(self._simulate)
        try:
            return self._assemble(control)
        except Exception:
            # If close fails, Python exception chaining preserves both failures.
            self.close(None)
            raise

    def _assemble(self, control: ControlOptions | None) -> GuiAssembly:
        from zcu_tools.gui.app.measure.state import State

        state = State(_make_empty_ctx())
        ctrl, window = self._build_window(
            state,
            self._registry,
            self._project_root,
        )

        self._controller = ctrl
        if self._environment is not None:
            ctrl.install_soc(self._environment.soc, self._environment.soccfg)

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
            window.remote_control_service = adapter

        return GuiAssembly(controller=ctrl, window=window, control_adapter=adapter)

    def close(self, assembly: GuiAssembly | None) -> None:
        """Drain app work before closing this environment's exact DeviceManager.

        Runtime calls on both startup failure and normal exit. Driver close
        failures retain registry entries and propagate; no hidden retry occurs.
        Controller.close only drains, it never closes drivers itself.
        """
        if assembly is not None:
            if assembly.control_adapter is not None:
                assembly.control_adapter.stop()
            ctrl = assembly.controller
            if _is_controller(ctrl):
                ctrl.close()
        if self._controller is not None:
            self._controller.close()
        if self._environment is not None and not self._environment_close_attempted:
            self._environment_close_attempted = True
            self._environment.devices.close_all_devices()

    def before_show(self, assembly: GuiAssembly) -> None:
        from qtpy.QtWidgets import QApplication

        from zcu_tools.gui.app.measure.services import create_persistence_caretaker

        window = assembly.window
        assert _is_main_window(window)
        application = QApplication.instance()
        if not isinstance(application, QApplication):
            raise RuntimeError("measure app requires a QApplication before assembly")
        # before_show precedes the runtime's remote stop connection on aboutToQuit.
        window.watch_application_shutdown(application)
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

    def _build_window(
        self,
        state: State,
        registry: Registry,
        project_root: str | None = None,
    ) -> tuple[Controller, MainWindow]:
        """Create Controller + MainWindow in the correct order."""

        from zcu_tools.gui.app.measure.controller import Controller
        from zcu_tools.gui.app.measure.ui.main_window import MainWindow
        from zcu_tools.gui.event_bus import BaseEventBus

        bus = BaseEventBus()

        ctrl = Controller(
            state=state,
            registry=registry,
            view=None,
            bus=bus,
            project_root=project_root,
            catalog_loader=self._catalog_loader,
            params_for=self._params_for,
            alias_presets=self.alias_presets,
            device_registry=None
            if self._environment is None
            else self._environment.devices,
            driver_factory=None
            if self._environment is None
            else self._environment.create_device,
        )

        try:
            window = MainWindow(
                ctrl,
                templates=lambda: registry.templates,
                alias_presets=self.alias_presets,
            )
            ctrl.add_view(window)
        except Exception:
            ctrl.close()
            raise
        return ctrl, window


def _is_controller(value: object) -> TypeGuard[Controller]:
    from zcu_tools.gui.app.measure.controller import Controller

    return isinstance(value, Controller)


def _is_main_window(value: object) -> TypeGuard[MainWindow]:
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    return isinstance(value, MainWindow)
