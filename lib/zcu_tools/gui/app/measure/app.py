"""GUI composition root — assembles all components for the shared runtime.

This module depends only on ``zcu_tools.gui.app.measure``; it does not know which concrete
experiments exist. The entry script wires a populated ``Registry`` with its injected templates
(built from caller-owned definitions) and passes it in — so the
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

    def assemble(self, control: ControlOptions | None) -> GuiAssembly:
        from zcu_tools.gui.app.measure.state import State

        state = State(_make_empty_ctx())
        ctrl, window = _build_window(
            state,
            self._registry,
            self._project_root,
            catalog_loader=self._catalog_loader,
            params_for=self._params_for,
            alias_presets=self.alias_presets,
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


def _is_controller(value: object) -> TypeGuard[Controller]:
    from zcu_tools.gui.app.measure.controller import Controller

    return isinstance(value, Controller)


def _is_main_window(value: object) -> TypeGuard[MainWindow]:
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    return isinstance(value, MainWindow)


def _build_window(
    state: State,
    registry: Registry,
    project_root: str | None = None,
    *,
    catalog_loader: ExperimentCatalogLoader | None = None,
    params_for: ParamsFor | None = None,
    alias_presets: AliasPresets | None = None,
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
        catalog_loader=catalog_loader,
        params_for=params_for,
        alias_presets=alias_presets,
    )

    window = MainWindow(ctrl, templates=lambda: registry.templates)
    ctrl.add_view(window)
    return ctrl, window
