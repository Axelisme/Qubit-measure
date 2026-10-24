"""Synthetic registered workflow host shared by component-driver scenarios."""

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from threading import Event

from pydantic import BaseModel, ConfigDict
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import Live, WorkflowEnv, workflow
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.workflow_attachment import WorkflowAttachment
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._synthetic import SampleAdapter


class Plan(FormModel):
    cycles: int = 1


class Tunables(FormModel):
    gain: float = 0.25


class NativePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    cycles: int


class NativeTunables(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    gain: float


@workflow("synthetic", "Synthetic retained-worker wait")
async def _waiting(
    env: WorkflowEnv[None], plan: NativePlan, tun: Live[NativeTunables]
) -> None:
    # A public wait keeps this invocation live until the scenario requests Stop.
    await env.wait_until(env.started_at + timedelta(days=1))


def workflow_window(
    qapp: QApplication, tmp_path: Path
) -> Iterator[tuple[Controller, MainWindow]]:
    """Yield an offline registered workflow window, then stop and drain its runs."""
    registry = Registry()
    registry.register("experiment", SampleAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
        progress_transport=DirectProgressTransport(),
    )
    yield from host_window(qapp, ctrl)


def host_window(
    qapp: QApplication, ctrl: Controller
) -> Iterator[tuple[Controller, MainWindow]]:
    """Mount synthetic workflows on ctrl with an offline board and owned teardown.

    qapp must be the live GUI-thread application. ctrl must be a fresh controller;
    this creates its entry/point, then closes it before deleting the view.
    Setup and teardown failures propagate.
    """
    ctrl.workflow_control().configure(
        (
            WorkflowAttachment(
                script=_waiting,
                plan_model=Plan,
                knobs_model=Tunables,
                plan_to_cfg=lambda form, env, snapshot: NativePlan(cycles=form.cycles),
                knobs_to_cfg=lambda form: NativeTunables(gain=form.gain),
                seed=lambda plan, form: None,
            ),
        )
    )
    ctrl.setup_control.new_entry(EntryRequest("workflow-ui"))
    ctrl.setup_control.new_point(PointCreateRequest("source", "", CommandOrigin("gui")))
    board = white_noise()
    ctrl.install_soc(board.soc, board.soccfg)
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    ctrl.add_view(window)
    window.refresh_context_panel()
    window.show()
    qapp.processEvents()
    destroyed = Event()
    window.destroyed.connect(destroyed.set)
    try:
        yield ctrl, window
    finally:
        # close() drains but does not cancel a retained Ask/wait after an assertion
        # failure. Release each live invocation before waiting for owner delivery.
        control = ctrl.workflow_control()
        for tab in control.tabs():
            if tab.busy and tab.run_id is not None:
                control.stop(tab.run_id)
        ctrl.close()
        if not destroyed.is_set():
            window.deleteLater()
        qapp.processEvents()
