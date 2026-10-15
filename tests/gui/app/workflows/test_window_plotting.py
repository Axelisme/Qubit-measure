"""Ordinary workflow Plots keep dense updates responsive and retain final pixels."""

from concurrent.futures import ThreadPoolExecutor
from threading import get_ident

import numpy as np
from matplotlib.backend_bases import DrawEvent
from matplotlib.figure import Figure
from numpy.typing import NDArray
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.workflows import Done, Step, WorkflowEnv, workflow
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.resources.entry import ResultEntry

from tests.gui._qt_waits import qt_until
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
        if not future.done():
            window.stop("dense")
            # Stop interrupts the accepted producer; keep serving its owner calls.
            qt_until(completed, label="stopped plot producer")
        executor.shutdown(wait=True)
        if harness.background.work is not None:
            harness.owner.pump_all()
            harness.background.deliver()
        qapp.processEvents()


def test_dense_updates_coalesce_on_owner_with_responsive_loop_and_latest_png(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
    qapp: QApplication,
) -> None:
    window, harness, _entry = attachment
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
        xs = np.array([0.0, 1.0])
        for value in range(80):
            viewer.update(xs, np.array([float(value), float(value + 1)]))
            updates.append(value)
            if value in (0, 40):
                # Hold this synthetic producer until a real owner turn acknowledges
                # each phase. Scheduling speed cannot determine responsiveness.
                harness.owner.call(
                    lambda: owner_turns.append((len(updates), get_ident()))
                )
        yield from ()
        return Done()

    catalog_source.install(step, window)
    assert isinstance(window.start("dense"), int)
    execute_plot_worker(qapp, window, harness)
    assert harness.background.error is None
    status = window.controller.status()
    assert status is not None and status.lifecycle == "done"
    assert len(updates) == 80
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
