"""Live committed overnight plots and independent read-only reopened views."""

from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

import numpy as np
import pytest
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.workflows import InvalidRunState, RunIdentity, RunPaths
from zcu_tools.experiment.workflows.artifacts import RunArtifacts
from zcu_tools.experiment.workflows.clock import UtcClock
from zcu_tools.experiment.workflows.encoding import encode_record
from zcu_tools.experiment.workflows.journal import RunMetadata, StepCommitted
from zcu_tools.gui.plotting.figure_export import render_figure_png

from tests.gui.app.workflows._start_fakes import StartHarness
from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.workflows.gui.window import T1WorkflowWindow
from zcu_lab.workflows.overnight import OvernightPoint

from ._sim_session import SimSession, wait_owner


def start_pairs(sim_session: SimSession, inputs: Inputs, run_id: str) -> None:
    """Start two simulator pairs with enough wait to observe the first commit."""
    window = sim_session.app.window
    window.select_workflow("overnight")
    result = window.start(
        run_id,
        plan_batch={
            ("flux_dev",): "flux",
            ("flux_unit",): "native",
            ("flux",): inputs.flux_value,
            ("outer_rounds",): 2,
            ("start_interval_s",): 1.0,
            ("points",): 21,
            ("qub_pulse", "kind"): "ref",
            ("qub_pulse", "name"): "Q1.pulses.pi",
            ("readout", "kind"): "ref",
            ("readout", "name"): "Q1.readout.ro",
        },
        tunables_batch={("reps",): 2000},
    )
    assert isinstance(result, int)


@pytest.mark.uses_wall_clock
def test_live_and_terminal_include_commits_and_reject_active_reopen(
    sim_session: SimSession, qapp: QApplication, inputs: Inputs
) -> None:
    window = sim_session.app.window
    start_pairs(sim_session, inputs, "trend-live")

    def first_commit() -> bool:
        status = window.controller.status()
        return status is not None and status.committed_seq == 1

    wait_owner(qapp, first_commit)
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)
    loop.exec()
    assert "overnight_trend" in window.figure_names("trend-live")
    first_png = window.figure_png("trend-live", "overnight_trend")
    root = sim_session.entry.result_path / "workflows/trend-live"
    with pytest.raises(InvalidRunState):
        window.open_overnight_trend(root)
    wait_owner(qapp, lambda: window.controller.operation_id is None)
    status = window.controller.status()
    assert status is not None and status.committed_seq == 2
    assert window.figure_png("trend-live", "overnight_trend") != first_png


@pytest.mark.uses_wall_clock
def test_fresh_window_reopens_same_plot_and_preserves_native_rpc_binding(
    sim_session: SimSession, qapp: QApplication, inputs: Inputs, tmp_path: Path
) -> None:
    window = sim_session.app.window
    start_pairs(sim_session, inputs, "trend-reopen")
    wait_owner(qapp, lambda: window.controller.operation_id is None)
    root = sim_session.entry.result_path / "workflows/trend-reopen"
    controller = window.controller
    reopened = window.open_overnight_trend(root)
    engine = controller.engine
    assert engine is not None
    assert len(engine.records("trend-reopen")) == 2
    before = window.figure_png("trend-reopen", "overnight_trend")
    with pytest.raises(FileNotFoundError):
        window.open_overnight_trend(tmp_path / "missing")
    assert window.controller is controller
    assert window.figure_png("trend-reopen", "overnight_trend") == before
    assert list(reopened) == ["overnight_trend"]
    assert render_figure_png(reopened["overnight_trend"]) == before
    clock = UtcClock()
    other = RunArtifacts(
        RunMetadata(
            RunIdentity(
                "different-source",
                SoftwareProvenance(
                    software_versions={},
                    git_commit=None,
                    git_dirty=None,
                    qick_version=None,
                    soc_fingerprint=None,
                    hostname="synthetic-test",
                ),
            ),
            "overnight",
            {},
            {},
            RunPaths(tmp_path / "other-meta", tmp_path / "other-data"),
            clock.now(),
        ),
        clock,
        RLock(),
    )
    point = OvernightPoint(
        0,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
        0.2,
        70.0,
        1.0,
        None,
        80.0,
        2.0,
        None,
    )
    other.append(StepCommitted(1, 1, 0, encode_record(point), "iter/000001", ()))
    other_figures = window.open_overnight_trend(other.paths.metadata_root)
    np.testing.assert_allclose(
        np.asarray(
            other_figures["overnight_trend"].axes[0].lines[0].get_ydata(), dtype=float
        ),
        [70.0],
    )
    assert window.figure_png("trend-reopen", "overnight_trend") == before
    harness = StartHarness()
    fresh = T1WorkflowWindow(harness.state, harness.services, harness.owner)
    try:
        figures = fresh.open_overnight_trend(root)
        assert fresh.controller.status() is None
        for reopened_axes, fresh_axes in zip(
            reopened["overnight_trend"].axes,
            figures["overnight_trend"].axes,
            strict=True,
        ):
            np.testing.assert_allclose(
                np.asarray(reopened_axes.lines[0].get_ydata(), dtype=float),
                np.asarray(fresh_axes.lines[0].get_ydata(), dtype=float),
                equal_nan=True,
            )
            assert (
                np.asarray(reopened_axes.lines[0].get_xdata()).tolist()
                == np.asarray(fresh_axes.lines[0].get_xdata()).tolist()
            )
            assert len(np.asarray(fresh_axes.lines[0].get_ydata())) == 2
    finally:
        assert fresh.close()
        fresh.deleteLater()
        qapp.processEvents()
