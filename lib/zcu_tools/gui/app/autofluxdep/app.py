"""Composition root for autofluxdep-gui.

``build_core`` assembles State + EventBus + Controller (the testable domain
core). ``AutoFluxDepGuiBehavior`` adds the Qt MainWindow behind the shared
process runtime.

The app composes the shared session services (connection / context / device /
startup) and uses the shared setup / device / predictor dialogs; the run drives
the orchestrator over the node graph, each node building its run cfg from the
active context and acquiring through either the selected hardware path or MockSoc.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeGuard

from zcu_tools.gui.remote.rpc_endpoint import ControlOptions
from zcu_tools.gui.runtime import (
    GuiAssembly,
    GuiRuntimeBehavior,
    GuiRuntimeSpec,
    PlotPolicy,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.controller import Controller
    from zcu_tools.gui.app.autofluxdep.state import ProjectInfo
    from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow


def _make_empty_ctx():
    """Minimal startup context: real empty MetaDict/ModuleLibrary, no file sync."""
    from zcu_tools.gui.session.types import ExpContext
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    return ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)


def build_core(
    project: ProjectInfo | None = None,
    project_root: str | None = None,
) -> Controller:
    """Wire State + EventBus + Controller — the testable domain core.

    ``project_root`` is the base directory default result/database paths anchor
    under (the setup dialog derives defaults against it). None falls back to cwd —
    fine for tests / a ``python -m`` run from the repo root; the launcher injects
    the repo root so the defaults are correct regardless of the working directory.
    """
    from zcu_tools.gui.app.autofluxdep.controller import Controller
    from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    state = AutoFluxDepState(_make_empty_ctx(), project=project)
    return Controller(state, EventBus(), project_root=project_root)


class AutoFluxDepGuiBehavior(GuiRuntimeBehavior):
    """autofluxdep-gui app wiring behind the shared GUI runtime."""

    spec: ClassVar[GuiRuntimeSpec] = GuiRuntimeSpec(
        app_name="autofluxdep",
        app_slug="autofluxdep",
        plot_policy=PlotPolicy.AGG_ONLY,
        default_control_port=8768,
        logging_extra_namespaces=("zcu_tools.program.v2",),
    )

    def __init__(
        self,
        project: ProjectInfo | None = None,
        *,
        project_root: str | None = None,
    ) -> None:
        self._project = project
        self._project_root = project_root or _repo_root()

    def assemble(self, control: ControlOptions | None) -> GuiAssembly:
        from zcu_tools.gui.app.autofluxdep.services import PersistenceCaretaker
        from zcu_tools.gui.app.autofluxdep.services.remote.service import (
            RemoteControlAdapter,
        )
        from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow

        ctrl = build_core(self._project, project_root=self._project_root)
        window = MainWindow(ctrl)

        caretaker = PersistenceCaretaker(ctrl)
        ctrl.attach_caretaker(caretaker)
        ctrl.restore_all()
        window.restore_workflow_view()

        adapter = (
            RemoteControlAdapter(ctrl, control, view=window)
            if control is not None
            else None
        )
        return GuiAssembly(controller=ctrl, window=window, control_adapter=adapter)

    def after_show(self, assembly: GuiAssembly) -> None:
        parent = assembly.window
        assert _is_main_window(parent)
        _show_startup_dialog(parent=parent)


def _repo_root() -> str:
    # This file is at <repo>/lib/zcu_tools/gui/app/autofluxdep/app.py.
    return str(Path(__file__).resolve().parents[5])


def _is_main_window(value: object) -> TypeGuard[MainWindow]:
    from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow

    return isinstance(value, MainWindow)


def _show_startup_dialog(parent: MainWindow) -> None:
    """Open the setup dialog non-modally on first launch.

    Mirrors ``zcu_tools.gui.app.main.app._show_startup_dialog``.  Non-modal is
    required so the Qt event loop keeps pumping while the dialog is visible —
    this lets background session operations (mock-soc connect, device setup)
    complete without deadlocking.
    """
    parent.open_setup_dialog(startup_mode=True)
