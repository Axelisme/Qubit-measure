"""Native buffer pressure must coalesce presentation without losing the last data."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from threading import get_ident
from time import monotonic

import numpy as np
import pytest
from matplotlib.backend_bases import DrawEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.plotting.plots import LinePlot
from zcu_tools.resources.entry import workflow_roots

from tests.zcu_lab.workflows._point_files import point_files
from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp

from ._sim_session import SimSession, wait_owner


@dataclass
class DisplayObservations:
    """Public live publications, rendered samples/thread IDs, and owner timer ticks."""

    publications: list[tuple[float, NDArray[np.float64]]] = field(default_factory=list)
    draws: list[tuple[int, NDArray[np.float64]]] = field(default_factory=list)
    ticks: list[float] = field(default_factory=list)

    def draw(self, event: DrawEvent) -> None:
        """Capture T1 samples actually rendered by the existing Qt canvas."""
        figure = event.canvas.figure
        if figure.axes and figure.axes[0].get_xlabel() == "Time (us)":
            ax = figure.axes[0]
            if ax.lines:
                self.draws.append(
                    (get_ident(), np.array(ax.lines[0].get_ydata(), dtype=np.float64))
                )


@pytest.fixture
def display_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[DisplayObservations]:
    """Observe public updates/draw_event without replacing acquisition or rendering."""
    observations = DisplayObservations()
    original_update = LinePlot.update
    original_init = FigureCanvasQTAgg.__init__

    def attach(self: FigureCanvasQTAgg, figure: Figure | None = None) -> None:
        original_init(self, figure)
        self.mpl_connect("draw_event", observations.draw)

    def update(
        self: LinePlot,
        xs: NDArray[np.float64],
        signals: NDArray[np.float64],
        title: str | None = None,
        *,
        refresh: bool = True,
    ) -> None:
        # The T1 grid starts at zero and has 21 points; calibration grids differ.
        if xs.size == 21 and xs[0] == 0.0:
            observations.publications.append(
                (monotonic(), np.array(signals, copy=True))
            )
        original_update(self, xs, signals, title, refresh=refresh)

    monkeypatch.setattr(FigureCanvasQTAgg, "__init__", attach)
    monkeypatch.setattr(LinePlot, "update", update)
    timer = QTimer()
    timer.setInterval(1)
    timer.timeout.connect(lambda: observations.ticks.append(monotonic()))
    timer.start()
    try:
        yield observations
    finally:
        timer.stop()


def test_native_dense_updates_coalesce_with_responsive_owner_and_latest_png(
    sim_session: SimSession,
    inputs: Inputs,
    qapp: QApplication,
    display_observations: DisplayObservations,
) -> None:
    window = sim_session.app.window
    owner_id = get_ident()
    observations = display_observations
    window.plan_panel.state.write(
        {("fluxes", "values"): [inputs.flux_value]}, window.plan_panel.env
    )
    window.tunables_panel.state.write(
        {
            ("cal", "span_mhz"): 2.0,
            ("cal", "rabi_start_factor"): 0.125,
            ("rounds",): 80,
        },
        window.tunables_panel.env,
    )
    token = window.start("dense")
    assert isinstance(token, int)
    wait_owner(qapp, lambda: window.controller.operation_id is None)
    status = window.controller.status()
    assert status is not None and status.lifecycle == "done", status
    updates, draws = observations.publications, observations.draws
    assert len(updates) >= 80, "Shipped acquisition must produce dense live updates"
    assert (
        sum(updates[0][0] < tick < updates[-1][0] for tick in observations.ticks) >= 2
    )
    assert {thread for thread, _samples in draws} == {owner_id}
    assert len(draws) < len(updates), "GUI must coalesce dense native buffer refreshes"
    engine = window.controller.engine
    assert engine is not None
    record = engine.records("dense")[0]
    roots = workflow_roots(
        result_path=sim_session.entry.result_path,
        database_path=sim_session.entry.database_path,
        run_id="dense",
    )
    source, _metadata = load_run_with_metadata(
        roots.database / record.iteration_dir / point_files(record)[-1],
        spec=T1Exp().persistence_spec(),
    )
    np.testing.assert_allclose(updates[-1][1], source.result.signals.real)
    np.testing.assert_allclose(draws[-1][1], source.result.signals.real)
    image = sim_session.image("dense", "t1/decay")
    assert image.images[0].data == window.figure_png("dense", "t1/decay")
    assert image.images[0].data.startswith(b"\x89PNG\r\n\x1a\n")
