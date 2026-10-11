"""GUI and generated MCP select and run the three native fluxdep attachments."""

import pytest
from qtpy.QtWidgets import QApplication, QComboBox
from zcu_tools.cfg_model import PulseSpec
from zcu_tools.datafile import load_run_data
from zcu_tools.gui.form import FormErrors

from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.workflows.gui.remote import FormWrite

from ._sim_session import SimSession, wait_owner


@pytest.mark.uses_wall_clock
@pytest.mark.parametrize("name", ["T2echo_fluxdep", "T2ramsey_fluxdep", "MIST_fluxdep"])
@pytest.mark.parametrize("starter", ["gui", "mcp"])
def test_selected_fluxdep_gui_and_mcp_run_on_the_fixed_controller(
    sim_session: SimSession, qapp: QApplication, inputs: Inputs, name: str, starter: str
) -> None:
    window = sim_session.app.window
    controller = window.controller
    if name == "MIST_fluxdep":
        context = sim_session.app.services.context
        target = context.read_context().target
        context.set_module(
            target,
            "Q1.pulses.mist",
            PulseSpec.model_validate(
                inputs.readout.modules.readout.pulse_cfg.model_dump(mode="python")
            ),
        )
        window.prepare_forms()
    # Selection is observable through the real app: no secondary registry/engine.
    window.select_workflow(name)
    selector = window.findChild(QComboBox, "workflow.select")
    assert selector is not None and selector.currentText() == name
    inspected = window.inspect_form("plan")
    assert not isinstance(inspected, FormErrors)
    plan: list[FormWrite] = [
        {"path": ["flux_dev"], "value": "flux"},
        {"path": ["flux_unit"], "value": "native"},
        {"path": ["fluxes", "kind"], "value": "list"},
        {
            "path": ["fluxes", "values"],
            "value": [inputs.flux_value, inputs.flux_value + 0.001],
        },
        {"path": ["pi_length_us"], "value": 0.4},
        {"path": ["qub_pulse", "kind"], "value": "ref"},
        {"path": ["qub_pulse", "name"], "value": "Q1.pulses.pi"},
        {"path": ["readout", "kind"], "value": "ref"},
        {"path": ["readout", "name"], "value": "Q1.readout.ro"},
    ]
    tunables: list[FormWrite] = [
        {"path": ["cal", "span_mhz"], "value": 30.0},
        {"path": ["cal", "rabi_start_factor"], "value": 0.125},
        {"path": ["rounds"], "value": 10},
    ]
    if name == "MIST_fluxdep":
        plan.extend(
            [
                {"path": ["gain_start"], "value": 0.0},
                {"path": ["gain_stop"], "value": 1.0},
                {"path": ["gain_points"], "value": 3},
                {"path": ["reset"], "value": None},
                {"path": ["mist_pulse", "kind"], "value": "ref"},
                {"path": ["mist_pulse", "name"], "value": "Q1.pulses.mist"},
            ]
        )
    else:
        plan.append({"path": ["points"], "value": 101})
        if name == "T2ramsey_fluxdep":
            tunables.append({"path": ["detune_ratio"], "value": 0.25})
    run_id = "20261006T010000Z-a1b2c3"
    if starter == "gui":
        window.plan_panel.state.write(
            {tuple(v["path"]): v["value"] for v in plan}, window.plan_panel.env
        )
        window.tunables_panel.state.write(
            {tuple(v["path"]): v["value"] for v in tunables}, window.tunables_panel.env
        )
        token = window.start(run_id)
    else:
        token = sim_session.call(
            "start",
            {"run_id": run_id, "name": name, "plan": plan, "tunables": tunables},
        )["operation_id"]
    assert isinstance(token, int)
    wait_owner(qapp, lambda: controller.operation_id is None)
    assert window.controller is controller
    status = controller.status()
    assert status is not None and status.lifecycle == "done", status
    assert status.workflow == name and status.committed_seq == 2
    engine = controller.engine
    assert engine is not None and len(engine.records(run_id)) == 2
    canonical = engine.records(run_id)
    report = window.export_fluxdep(run_id)
    assert all(output.status == "saved" for output in report.outputs), report
    native_path = report.outputs[0].path
    assert native_path is not None
    merged = load_run_data(native_path)
    assert merged.metadata.experiment == name
    assert merged.metadata.snapshot.point == "source"
    assert engine.records(run_id) == canonical
    assert tuple(merged.payload.variables) == (
        ("signal", "success")
        if name == "MIST_fluxdep"
        else ("signal", "time", name.split("_")[0].lower())
    )
