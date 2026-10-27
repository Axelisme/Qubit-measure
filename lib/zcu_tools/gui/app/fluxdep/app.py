"""Composition root for the fluxdep-gui.

Wires State + Controller + MainWindow behind the shared GUI process runtime.
Unlike measure-gui there is no registry/role-catalog (no experiment adapters)
and no session-restore caretaker. The tool starts by loading raw spectra or a
spectrums v2 checkpoint. Export writes an eligible predictor v2, declared setup
fields and the entry analysis ledger.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from zcu_tools.gui.app.analysis_project.flux import ComponentFluxProvider
from zcu_tools.gui.app.fluxdep.export_plan import ExportPlanner
from zcu_tools.gui.runtime import (
    GuiAssembly,
    GuiRuntimeBehavior,
    GuiRuntimeSpec,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
    from zcu_tools.gui.remote.rpc_endpoint import ControlOptions


class FluxDepGuiBehavior(GuiRuntimeBehavior):
    spec = GuiRuntimeSpec(
        app_name="fluxdep",
        app_slug="fluxdep",
        default_control_port=8766,
    )

    def __init__(
        self,
        project: AnalysisTarget | None = None,
        project_root: str | None = None,
        export_planner: ExportPlanner | None = None,
        flux_context_provider: ComponentFluxProvider | None = None,
    ) -> None:
        """Bind entry/path context and the lab's component export policy.

        export_planner receives detached setup and native reference positions.
        When omitted, viewing/search remain available but export is refused.
        """
        self._export_planner = export_planner
        self._flux_context_provider = flux_context_provider
        self._project = project
        self._project_root = project_root

    def assemble(self, control: ControlOptions | None) -> GuiAssembly:
        from zcu_tools.gui.app.fluxdep.controller import Controller
        from zcu_tools.gui.app.fluxdep.event_bus import EventBus
        from zcu_tools.gui.app.fluxdep.remote.service import (
            RemoteControlAdapter,
        )
        from zcu_tools.gui.app.fluxdep.search import FluxDepSearchRuntime
        from zcu_tools.gui.app.fluxdep.state import FluxDepState
        from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
        from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
        from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
        from zcu_tools.gui.session.adapters.qt_progress_transport import (
            QtProgressTransport,
        )
        from zcu_tools.gui.session.services.progress import ProgressService

        owner = QtOwnerScheduler()
        runner = BackgroundRunner()
        search_runner = BackgroundRunner()
        search_transport = QtProgressTransport()
        search_progress = ProgressService(search_transport)

        def submit_interactive(
            compute: Callable[[], object],
            on_done: Callable[[object], None],
            on_error: Callable[[Exception], None],
        ) -> None:
            runner.submit(compute, on_done=on_done, on_error=on_error)

        ctrl = Controller(
            FluxDepState(self._project),
            EventBus(),
            project_root=self._project_root,
            interactive_owner=owner,
            interactive_background=submit_interactive,
            search_runtime=FluxDepSearchRuntime(search_runner, search_progress),
        )
        if self._flux_context_provider is not None:
            ctrl.configure_flux_context(self._flux_context_provider)
        if self._export_planner is not None:
            ctrl.configure_export(self._export_planner)
        window = MainWindow(
            ctrl, interactive_runner=runner, search_runner=search_runner
        )
        search_transport.setParent(window)
        adapter = (
            RemoteControlAdapter(
                ctrl,
                control,
                owner_scheduler=owner,
            )
            if control is not None
            else None
        )
        return GuiAssembly(controller=ctrl, window=window, control_adapter=adapter)

    def before_show(self, assembly: GuiAssembly) -> None:
        """Retire app-local authority views before normal Qt exit stops remote."""
        from qtpy.QtWidgets import QApplication

        from .ui.main_window import MainWindow

        window = assembly.window
        if not isinstance(window, MainWindow):
            raise TypeError("fluxdep assembly requires its MainWindow")
        application = QApplication.instance()
        if not isinstance(application, QApplication):
            raise RuntimeError("fluxdep app requires a QApplication before assembly")
        window.watch_application_shutdown(application)


__all__ = ["FluxDepGuiBehavior"]
