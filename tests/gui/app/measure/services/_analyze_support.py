"""Per-case builders and recording executor shared by AnalyzeService owners."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

from zcu_tools.gui.app.measure.adapter import ContextReadiness
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui._progress_fakes import DirectProgressTransport


def make_plots() -> Plots:
    """Return fresh plots with a non-presenting host for one test case."""
    return Plots(NonPresentingHost())


def make_state(tab_id: str = "tab1") -> State:
    """Create an active offline state with one mock tab and a fake run result."""
    ctx = SessionEnv(
        parameters=None,
        soc=MagicMock(),
        soccfg=MagicMock(),
        result_dir="/tmp",
        readiness=ContextReadiness.ACTIVE,
    )
    state = State(ctx)
    state.add_tab(
        tab_id,
        Session(adapter_name="fake", adapter=MagicMock(), cfg=MagicMock()),
    )
    # Provide a fake run_result so the tab is not empty
    state.update_tab_result(tab_id, object())
    return state


class FakeBg:
    """Capture submitted work and callbacks without executing or scheduling them.

    Each instance starts with no last_work, last_on_done or last_on_error and
    submit_count zero. fail_submit rejects submissions with RuntimeError instead
    of recording them; otherwise each submission replaces the captured callbacks.
    """

    def __init__(self, *, fail_submit: bool = False) -> None:
        self._fail_submit = fail_submit
        self.last_work: Callable[[], Any] | None = None
        self.last_on_done: Callable[[Any], None] | None = None
        self.last_on_error: Callable[[Exception], None] | None = None
        self.submit_count = 0

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Record work and completion callbacks, ignoring the pool selection.

        Raise RuntimeError without recording when fail_submit was requested.
        The test owns invoking the captured work and its done or error callback.
        """
        if self._fail_submit:
            raise RuntimeError("submit boom")
        self.submit_count += 1
        self.last_work = work
        self.last_on_done = on_done
        self.last_on_error = on_error


def make_service(
    state: State,
    bus: EventBus,
    *,
    fail_submit: bool = False,
    handles: OperationHandles | None = None,
    writeback: MagicMock | None = None,
) -> tuple[AnalyzeService, FakeBg]:
    """Wire a fresh service and recording executor to the supplied state and bus.

    Use supplied handles and writeback mocks when present, or create fresh ones.
    Reset create_draft's return value to None. fail_submit makes the executor
    reject submissions with RuntimeError; no real background work is started.
    """
    bg = FakeBg(fail_submit=fail_submit)
    handles = handles or OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    writeback = writeback if writeback is not None else MagicMock()
    writeback.create_draft.return_value = None
    runner = OperationRunner(MagicMock(), handles, progress, bg, bus)
    svc = AnalyzeService(state, runner, bus, writeback, handles)
    return svc, bg


def make_two_tab_state() -> State:
    """Create a fresh active state with independent tab1/tab2 mock run results."""
    state = make_state("tab1")
    state.add_tab(
        "tab2",
        Session(adapter_name="fake", adapter=MagicMock(), cfg=MagicMock()),
    )
    state.update_tab_result("tab2", object())
    return state
