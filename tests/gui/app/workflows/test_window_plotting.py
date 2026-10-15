"""Ordinary workflow Plots keep dense updates responsive and retain final pixels."""

from concurrent.futures import ThreadPoolExecutor
from threading import get_ident

import numpy as np
import pytest
from matplotlib.backend_bases import DrawEvent
from matplotlib.figure import Figure
from numpy.typing import NDArray
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.workflows import Done, Step, WorkflowEnv, workflow
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.resources.entry import ResultEntry

from tests.gui._qt_waits import qt_until
from tests.gui.app._owner_waits import owner_result
from tests.gui.app.workflows._reload_source import InjectedCatalog
from tests.gui.app.workflows._start_fakes import StartHarness
from tests.gui.app.workflows._synthetic import Context, Plan, Tunables, init_sample


def execute_plot_worker(
    qapp: QApplication, window: WorkflowWindow[Context], harness: StartHarness
) -> None:
    """Pump real Qt and injected owner calls until the submitted Engine returns."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(harness.background.execute)

    def completed() -> bool:
        harness.owner.pump_once()
        return future.done()

    try:
        # Submitted Engine work, including plot dispatch, returns after owner replies.
        qt_until(completed, label="dense plot producer")
        future.result()
    finally:
        try:
            if not future.done():
                window.stop("dense")
                try:
                    # Stop interrupts the accepted producer; serve its owner calls.
                    qt_until(completed, label="stopped plot producer")
                finally:
                    # Even a failed Qt wait must service outstanding synchronous calls
                    # before joining this producer. The original wait error propagates.
                    owner_result(harness.owner, future, label="plot cleanup join")
        finally:
            executor.shutdown(wait=True)
            if harness.background.work is not None:
                harness.owner.pump_all()
                harness.background.deliver()
            qapp.processEvents()


class RefreshDelivery:
    """Hold no-argument starts of single-shot 50 ms QTimers for manual delivery.

    Other starts retain real Qt scheduling. Held timers report active until stopped
    or delivered; stopping removes them from pending.
    Pass a pytest monkeypatch fixture or scoped MonkeyPatch to own the QTimer
    method patches. Keep it active while using this helper, then undo it; fixture
    teardown or the MonkeyPatch context exit restores the original methods.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Distinct held timers awaiting delivery; empty when none remain pending.
        self.pending: set[QTimer] = set()
        # Ordered intercepted starts, including repeats; retained after stop/delivery.
        self.starts: list[QTimer] = []
        original_start, original_stop, original_active = (
            QTimer.start,
            QTimer.stop,
            QTimer.isActive,
        )

        def start(timer: QTimer, msec: int | None = None) -> None:
            # Select the documented refresh policy, not host state or timer registries.
            # Probe/watchdog and all unrelated timers retain their real Qt scheduling.
            if msec is None and timer.isSingleShot() and timer.interval() == 50:
                self.pending.add(timer)
                self.starts.append(timer)
            elif msec is None:
                original_start(timer)
            else:
                original_start(timer, msec)

        def stop(timer: QTimer) -> None:
            self.pending.discard(timer)
            original_stop(timer)

        def active(timer: QTimer) -> bool:
            return timer in self.pending or original_active(timer)

        monkeypatch.setattr(QTimer, "start", start)
        monkeypatch.setattr(QTimer, "stop", stop)
        monkeypatch.setattr(QTimer, "isActive", active)

    def deliver(self) -> None:
        """Remove the sole pending timer and emit its native timeout signal.

        Raise AssertionError unless exactly one timer is pending.
        """
        assert len(self.pending) == 1
        timer = self.pending.pop()
        timer.timeout.emit()


def test_dense_updates_coalesce_on_owner_with_responsive_loop_and_latest_png(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, _entry = attachment
    refresh = RefreshDelivery(monkeypatch)
    owner_id = get_ident()
    updates: list[int] = []
    owner_turns: list[tuple[int, int]] = []
    draws: list[tuple[int, NDArray[np.float64]]] = []

    def drawn(event: DrawEvent) -> None:
        line = event.canvas.figure.axes[0].lines[0]
        draws.append((get_ident(), np.array(line.get_ydata(), dtype=np.float64)))

    @workflow("dense", plan=Plan, tunables=Tunables, init=init_sample)
    def step(
        env: WorkflowEnv[Context], plan: Plan, tun: Tunables, state: int
    ) -> Step[int, float]:
        del plan, tun, state
        viewer = env.plots.liveplot_1d("ordinary", "Index", "Value")

        def observe(figure: Figure) -> None:
            figure.canvas.mpl_connect("draw_event", drawn)

        env.plots.draw("ordinary", observe)

        def acknowledge() -> None:
            owner_turns.append((len(updates), get_ident()))
            assert len(refresh.starts) == 1
            if len(updates) == 41:
                # All 41 real updates precede the first external timeout delivery.
                refresh.deliver()

        xs = np.array([0.0, 1.0])
        for value in range(80):
            viewer.update(xs, np.array([float(value), float(value + 1)]))
            updates.append(value)
            if value in (0, 40):
                # Hold this synthetic producer until a real owner turn acknowledges
                # each phase. Scheduling speed cannot determine responsiveness.
                harness.owner.call(acknowledge)
        yield from ()
        return Done()

    catalog_source.install(step, window)
    assert isinstance(window.start("dense"), int)
    execute_plot_worker(qapp, window, harness)
    assert harness.background.error is None
    status = window.controller.status()
    assert status is not None and status.lifecycle == "done"
    assert len(updates) == 80
    assert len(refresh.starts) == 2
    assert not refresh.pending, "Final refresh must cancel the remaining timeout"
    assert owner_turns == [(1, owner_id), (41, owner_id)]
    assert {thread for thread, _values in draws} == {owner_id}
    assert len(draws) < len(updates), "Dense refreshes must coalesce canvas draws"
    np.testing.assert_array_equal(draws[-1][1], [79.0, 80.0])
    assert window.figure_names("dense") == ("ordinary",)
    png = window.figure_png("dense", "ordinary")
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert draws[-1][0] == owner_id
    np.testing.assert_array_equal(draws[-1][1], [79.0, 80.0])
    assert window.figure_png("dense", "ordinary") == png
