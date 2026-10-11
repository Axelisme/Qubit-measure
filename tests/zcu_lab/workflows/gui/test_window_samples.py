"""Manual GUI/MCP exports retain accepted origin and native device evidence."""

import pytest
from qtpy.QtWidgets import QLabel, QPushButton
from zcu_tools.datafile import load_run_data
from zcu_tools.device.fake import FakeDevice
from zcu_tools.gui.form import FormEnv, FormInput
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.resources.entry import ResultEntry

from tests.gui.app.workflows._start_fakes import StartHarness
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.gui.window import T1WorkflowWindow

from ._recording_transport import WindowTransport, WireRoute


def test_manual_export_keeps_accepted_origin_after_source_and_selection_change(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    mcp_tools: ToolTable,
    mcp_transport: WindowTransport,
) -> None:
    window, harness, entry = attachment
    button = window.findChild(QPushButton, "workflow.run.export_samples")
    assert button is not None, "Window must offer a manual samples export action"
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
        window.export_samples(run_id)
    harness.complete_segment()
    status = window.controller.status()
    assert status is not None and status.lifecycle == "done"
    table = entry.samples("Q1")
    assert not table.path.exists(), "Terminal delivery must not auto-export samples"
    engine = window.controller.engine
    assert engine is not None
    record = engine.records(run_id)[0]
    native = load_run_data(
        entry.database_path
        / "workflows"
        / run_id
        / record.iteration_dir
        / record.run_files[-1]
    )
    assert native.metadata.device_snapshot is not None
    value = native.metadata.device_snapshot["flux"]["value"]
    driver = harness.services.device.get_connected_devices()["flux"]
    assert isinstance(driver, FakeDevice)
    driver.set_value(777)
    entry.new_point("later")
    harness.services.context.use_point(PointUseRequest("later"))
    window.select_workflow("overnight")
    assert button.isEnabled(), (
        "Current draft must not replace an accepted run's export capability"
    )
    events = entry.ledger.events()
    button.click()
    row = table.latest(run_id + "/0")
    assert (
        row is not None
        and row["point"] == "source"
        and row["source"] == "workflow:" + run_id
    )
    assert (
        row["dev_value"] == value
        and row["dev_value"] != 777
        and row["dev_unit"] == "native"
    )
    assert row["t1_us"] > 0 and row["t1_err_us"] >= 0 and row["freq_mhz"] > 0
    assert entry.ledger.events() == events
    before = len(mcp_transport.requests)
    reply = mcp_tools["workflows_workflow_export_samples"]["handler"](
        {"run_id": run_id}
    )
    assert len(mcp_transport.requests) == before + 1
    assert (
        reply["written_count"] == 1
        and reply["run_id"] == run_id
        and reply["path"] == str(table.path)
    )
    assert len(table.path.read_text().splitlines()) == 2 and len(table.read()) == 1
    with pytest.raises(RuntimeError, match="run_mismatch"):
        mcp_tools["workflows_workflow_export_samples"]["handler"]({"run_id": "stale"})
    assert window.controller.status() == status


def test_gui_and_wire_report_zero_prefix_io_failure(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    wire_route: WireRoute,
) -> None:
    window, harness, entry = attachment
    button = window.findChild(QPushButton, "workflow.run.export_samples")
    assert button is not None, (
        "Sample failures must be observable through the GUI and wire"
    )
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
    button.click()
    message = window.findChild(QLabel, "workflow.run.message")
    assert (
        message is not None
        and "0 rows" in message.text()
        and str(table.path) in message.text()
    )
    response = wire_route(
        Request("sample-fail", "workflow.export_samples", {"run_id": run_id})
    )
    assert response["ok"] is False
    error = response["error"]
    assert (
        isinstance(error, dict)
        and error["code"] == "controller_error"
        and error["reason"] == "sample_export_failed"
    )
    assert (
        error["data"]["written_count"] == 0
        and error["data"]["cause_type"] == "IsADirectoryError"
    )
    assert str(table.path) in error["data"]["cause"]


@pytest.mark.parametrize(
    "name, reason", [("MIST_fluxdep", "science"), ("overnight", "flux scan")]
)
def test_known_unsupported_attachment_explains_disabled_action(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    name: str,
    reason: str,
) -> None:
    window, _harness, _entry = attachment
    window.select_workflow(name)
    button = window.findChild(QPushButton, "workflow.run.export_samples")
    assert button is not None, (
        "Unsupported workflows must expose a disabled, explained export"
    )
    assert not button.isEnabled() and reason in button.toolTip()
