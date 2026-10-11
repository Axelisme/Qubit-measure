"""The single Window selector, GUI and MCP run the same current native workflow."""

import pytest
from qtpy.QtWidgets import QApplication, QComboBox
from zcu_tools.gui.form import FormErrors

from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.workflows.gui.remote import FormWrite

from ._sim_session import SimSession, wait_owner


def overnight_plan(inputs: Inputs) -> list[FormWrite]:
    """Supply fixed native module refs and explicit monitoring parameters."""
    return [
        {"path": ["flux_dev"], "value": "flux"},
        {"path": ["flux_unit"], "value": "native"},
        {"path": ["flux"], "value": inputs.flux_value},
        {"path": ["outer_rounds"], "value": 2},
        {"path": ["start_interval_s"], "value": 0.01},
        {"path": ["points"], "value": 21},
        {"path": ["qub_pulse", "kind"], "value": "ref"},
        {"path": ["qub_pulse", "name"], "value": "Q1.pulses.pi"},
        {"path": ["readout", "kind"], "value": "ref"},
        {"path": ["readout", "name"], "value": "Q1.readout.ro"},
    ]


@pytest.mark.uses_wall_clock
@pytest.mark.parametrize("starter", ["gui", "mcp"])
def test_selected_overnight_gui_and_mcp_commit_same_native_pairs(
    sim_session: SimSession,
    qapp: QApplication,
    inputs: Inputs,
    starter: str,
) -> None:
    window = sim_session.app.window
    controller = window.controller
    window.select_workflow("overnight")
    selector = window.findChild(QComboBox, "workflow.select")
    assert selector is not None and selector.currentText() == "overnight"
    inspected = window.inspect_form("plan")
    assert not isinstance(inspected, FormErrors)
    properties = inspected["schema"]["properties"]
    assert isinstance(properties, dict)
    assert "outer_rounds" in properties and "pi_length_us" not in properties
    plan = overnight_plan(inputs)
    tunables: list[FormWrite] = [{"path": ["reps"], "value": 2000}]
    if starter == "gui":
        window.plan_panel.state.write(
            {tuple(item["path"]): item["value"] for item in plan},
            window.plan_panel.env,
        )
        window.tunables_panel.state.write(
            {tuple(item["path"]): item["value"] for item in tunables},
            window.tunables_panel.env,
        )
        token = window.start("overnight-host")
    else:
        token = sim_session.call(
            "start",
            {
                "run_id": "overnight-host",
                "name": "overnight",
                "plan": plan,
                "tunables": tunables,
            },
        )["operation_id"]
    assert isinstance(token, int)
    wait_owner(qapp, lambda: controller.operation_id is None)
    assert window.controller is controller
    status = controller.status()
    assert status is not None and status.lifecycle == "done", status
    assert status.workflow == "overnight" and status.committed_seq == 2
    wire = sim_session.call("status", {})
    wire_status = wire["status"]
    assert isinstance(wire_status, dict)
    assert wire_status["workflow"] == "overnight"
    assert wire_status["committed_seq"] == 2
    engine = controller.engine
    assert engine is not None
    records = engine.records("overnight-host")
    assert len(records) == 2 and all(len(record.run_files) == 2 for record in records)
    with pytest.raises(ValueError, match="fluxdep export binding"):
        window.export_fluxdep("overnight-host")


def test_explicit_unknown_mcp_start_keeps_selection_and_drafts(
    sim_session: SimSession,
) -> None:
    window = sim_session.app.window
    window.select_workflow("overnight")
    before = window.capture_persisted_state()
    with pytest.raises(RuntimeError, match="invalid_params.*unknown_workflow"):
        sim_session.call(
            "start",
            {"run_id": "unknown", "name": "missing-workflow", "plan": [], "tunables": []},
        )
    assert window.workflow_name == "overnight"
    assert window.capture_persisted_state() == before
    assert window.controller.run_id is None


def test_switching_workflows_preserves_separate_parsed_drafts(
    sim_session: SimSession,
) -> None:
    window = sim_session.app.window
    window.plan_panel.state.write(
        {("pi_length_us",): {"expr": "1 / 3"}}, window.plan_panel.env
    )
    window.select_workflow("overnight")
    window.plan_panel.state.write({("outer_rounds",): 7}, window.plan_panel.env)
    window.tunables_panel.state.write(
        {("t2_guess_us",): {"expr": "3 * 4"}}, window.tunables_panel.env
    )
    window.select_workflow("T1_fluxdep")
    assert window.capture_persisted_state().plan["pi_length_us"] == {"expr": "1 / 3"}
    window.select_workflow("overnight")
    saved = window.capture_persisted_state()
    assert saved.plan["outer_rounds"] == 7
    assert saved.knobs["t2_guess_us"] == {"expr": "3 * 4"}
