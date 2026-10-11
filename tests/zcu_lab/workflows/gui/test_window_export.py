"""Explicit terminal export through the public GUI and MCP routes."""

import pytest
from qtpy.QtWidgets import QLabel, QPushButton
from zcu_tools.gui.form import FormEnv, FormInput
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.resources.entry import ResultEntry

from tests.gui.app.workflows._start_fakes import StartHarness
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.gui.window import T1WorkflowWindow

from ._window_support import WindowTransport


def test_terminal_export_action_and_mcp_share_native_paths_and_ledger(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    mcp_tools: ToolTable,
    mcp_transport: WindowTransport,
) -> None:
    """Export is explicit and each frontend reports its own output attempts."""
    import os

    from zcu_tools.datafile import DataVariable, load_grouped_labber_data, load_run_data
    from zcu_tools.resources.entry import SavedPayload

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
    with pytest.raises(RuntimeError):
        window.export_fluxdep(run_id)
    harness.complete_segment()
    status = window.controller.status()
    assert status is not None and status.lifecycle == "done"
    merged = entry.database_path / "workflows" / run_id / "merged.h5"
    assert not merged.exists(), "terminal delivery must not automatically export"
    button = window.findChild(QPushButton, "workflow.run.export_fluxdep")
    assert button is not None and button.isEnabled()
    button.click()
    native = load_run_data(merged)
    assert native.metadata.snapshot.point == "source"
    assert native.payload.variables[DataVariable("signal")].z.shape[0] == 1
    message = window.findChild(QLabel, "workflow.run.message")
    assert (
        message is not None
        and "saved" in message.text()
        and str(merged) in message.text()
    )
    events = [event for event in entry.ledger.events() if event.kind == "saved"]
    assert len(events) == 1 and events[0].origin == "gui"
    first_payload = events[0].payload
    assert isinstance(first_payload, SavedPayload)
    before = len(mcp_transport.requests)
    exported = mcp_tools["workflows_workflow_export_fluxdep"]["handler"](
        {"run_id": run_id}
    )
    assert len(mcp_transport.requests) == before + 1
    assert exported["run_id"] == run_id and exported["ledger_error"] is None
    assert [item["status"] for item in exported["outputs"]] == ["saved", "saved"]
    second_payload = entry.ledger.events()[-1].payload
    assert isinstance(second_payload, SavedPayload)
    assert entry.ledger.events()[-1].origin == "mcp"
    assert (
        second_payload.labber_path
        and second_payload.labber_path != first_payload.labber_path
    )
    labber = load_grouped_labber_data(second_payload.labber_path)
    for key, original in native.payload.variables.items():
        import numpy as np

        np.testing.assert_array_equal(labber.variables[key].z, original.z)
    with pytest.raises(RuntimeError, match="run_mismatch"):
        mcp_tools["workflows_workflow_export_fluxdep"]["handler"]({"run_id": "stale"})
    screenshot = os.environ.get("WORKFLOW_EXPORT_SCREENSHOT")
    if screenshot:
        window.resize(900, 1000)
        window.ensurePolished()
        assert window.grab().save(screenshot)


def test_gui_start_action_creates_identity_accepted_by_native_export(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    """The ordinary Start button must not create a native-incompatible run ID."""
    from zcu_tools.datafile import load_run_data

    window, harness, entry = attachment
    window.plan_panel.state.write(
        {("fluxes", "kind"): "list", ("fluxes", "values"): [1.0]}, FormEnv()
    )
    window.tunables_panel.state.reset(
        FormInput(
            T1TunablesForm,
            {"cal": {"span_mhz": 2.0, "rabi_start_factor": 0.125}, "rounds": 2},
        )
    )
    start = window.findChild(QPushButton, "workflow.run.start")
    assert start is not None
    start.click()
    harness.complete_segment()
    run_id = window.controller.run_id
    assert run_id is not None
    export = window.findChild(QPushButton, "workflow.run.export_fluxdep")
    assert export is not None and export.isEnabled()
    export.click()
    message = window.findChild(QLabel, "workflow.run.message")
    assert message is not None and "data_h5: saved" in message.text()
    native = load_run_data(entry.database_path / "workflows" / run_id / "merged.h5")
    assert native.metadata.run_id == run_id
    assert native.metadata.snapshot.point == "source"
    status = window.controller.status()
    assert status is not None and status.lifecycle == "done"
