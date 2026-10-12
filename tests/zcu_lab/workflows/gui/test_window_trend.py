"""Live committed overnight plots through native terminal delivery."""

import pytest
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QApplication

from tests.zcu_lab.workflows._sim_inputs import Inputs

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
def test_live_and_terminal_include_commits(
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
    wait_owner(qapp, lambda: window.controller.operation_id is None)
    status = window.controller.status()
    assert status is not None and status.committed_seq == 2
    assert window.figure_png("trend-live", "overnight_trend") != first_png
