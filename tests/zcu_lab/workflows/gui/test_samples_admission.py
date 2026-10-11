"""Samples readiness follows native acceptance AND retained host delivery."""

import pytest
from qtpy.QtWidgets import QPushButton
from zcu_tools.experiment.workflows import InvalidRunState
from zcu_tools.gui.form import FormEnv, FormInput
from zcu_tools.gui.remote.wire import Request
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.resources.entry import ResultEntry

from tests.gui.app.workflows._start_fakes import StartHarness
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.gui.window import T1WorkflowWindow

from ._recording_transport import WireRoute


def test_samples_ready_waits_for_host_delivery(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, _entry = attachment
    run_id = "20261006T010000Z-a1b2c3"
    window.plan_panel.state.write(
        {("fluxes", "kind"): "list", ("fluxes", "values"): [1.0]}, FormEnv()
    )
    window.tunables_panel.state.reset(
        FormInput(
            T1TunablesForm,
            {"cal": {"span_mhz": 2.0, "rabi_start_factor": 0.125}, "rounds": 2},
        )
    )
    assert isinstance(window.start(run_id), int)
    assert not getattr(window.controller, "samples_export_ready", False)
    harness.background.execute()
    status = window.controller.status()
    assert status is not None and status.lifecycle == "done"
    assert not getattr(window.controller, "samples_export_ready", False)
    with pytest.raises(InvalidRunState, match="export samples"):
        window.export_samples(run_id)
    button = window.findChild(QPushButton, "workflow.run.export_samples")
    assert button is not None and not button.isEnabled()
    harness.background.deliver()
    assert getattr(window.controller, "samples_export_ready", False), (
        "Successful delivery must expose the accepted samples capability"
    )
    assert button.isEnabled()


def test_mcp_failure_retains_original_io_and_prefix_count(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry], mcp_tools: ToolTable
) -> None:
    window, harness, entry = attachment
    run_id = "20261006T010000Z-a1b2c3"
    window.plan_panel.state.write(
        {("fluxes", "kind"): "list", ("fluxes", "values"): [1.0]}, FormEnv()
    )
    window.tunables_panel.state.reset(
        FormInput(
            T1TunablesForm,
            {"cal": {"span_mhz": 2.0, "rabi_start_factor": 0.125}, "rounds": 2},
        )
    )
    assert isinstance(window.start(run_id), int)
    harness.complete_segment()
    table = entry.samples("Q1")
    table.path.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="sample_export_failed") as failure:
        mcp_tools["workflows_workflow_export_samples"]["handler"]({"run_id": run_id})
    assert (
        "0 rows" in str(failure.value)
        and str(table.path) in str(failure.value)
        and "Is a directory" in str(failure.value)
    )


def test_terminal_overnight_is_unsupported_not_zero_row_success(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    wire_route: WireRoute,
    mcp_tools: ToolTable,
) -> None:
    window, harness, entry = attachment
    run_id = "unsupported"
    result = window.start(
        run_id,
        name="overnight",
        plan_batch={
            ("flux_dev",): "flux",
            ("flux_unit",): "native",
            ("flux",): 1.0,
            ("outer_rounds",): 1,
            ("start_interval_s",): 1.0,
            ("points",): 21,
            ("qub_pulse", "kind"): "ref",
            ("qub_pulse", "name"): "Q1.pulses.pi",
            ("readout", "kind"): "ref",
            ("readout", "name"): "Q1.readout.ro",
        },
        tunables_batch={},
    )
    assert isinstance(result, int)
    harness.complete_segment()
    button = window.findChild(QPushButton, "workflow.run.export_samples")
    assert (
        button is not None
        and not button.isEnabled()
        and "flux scan" in button.toolTip()
    )
    response = wire_route(
        Request("unsupported", "workflow.export_samples", {"run_id": run_id})
    )
    assert response["ok"] is False
    error = response["error"]
    assert (
        isinstance(error, dict)
        and error["code"] == "precondition_failed"
        and error["reason"] == "unsupported"
    )
    with pytest.raises(RuntimeError, match="unsupported"):
        mcp_tools["workflows_workflow_export_samples"]["handler"]({"run_id": run_id})
    assert not entry.samples("Q1").path.exists()
