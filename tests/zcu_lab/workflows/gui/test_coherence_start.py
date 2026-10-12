"""Coherence uses the existing single Window/Controller GUI and MCP path."""

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import load_run_data

from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.workflows.gui.remote import FormWrite

from ._sim_session import SimSession, wait_owner
from ._window_support import start_plan_batch


@pytest.mark.uses_wall_clock
@pytest.mark.parametrize("starter", ["gui", "mcp"])
def test_coherence_flags_start_through_same_controller_and_native_records(
    sim_session: SimSession, qapp: QApplication, inputs: Inputs, starter: str
) -> None:
    window = sim_session.app.window
    controller = window.controller
    plan = start_plan_batch(inputs) + [
        {"path": ["measure_t1"], "value": False},
        {"path": ["measure_t2ramsey"], "value": False},
        {"path": ["measure_t2echo"], "value": True},
    ]
    knobs: list[FormWrite] = [
        {"path": ["cal", "span_mhz"], "value": 2.0},
        {"path": ["cal", "rabi_start_factor"], "value": 0.125},
        {"path": ["cal", "min_readout_snr"], "value": 1e12},
        {"path": ["t1", "guess_us"], "value": 50.0},
        {"path": ["t2ramsey", "detune_ratio"], "value": 0.25},
        {"path": ["t2echo", "rounds"], "value": 2},
    ]
    if starter == "gui":
        window.select_workflow("coherence_fluxdep")
        window.plan_panel.state.write(
            {tuple(item["path"]): item["value"] for item in plan}, window.plan_panel.env
        )
        window.tunables_panel.state.write(
            {tuple(item["path"]): item["value"] for item in knobs},
            window.tunables_panel.env,
        )
        token = window.start("coherence-host")
    else:
        token = sim_session.call(
            "start",
            {
                "run_id": "coherence-host",
                "name": "coherence_fluxdep",
                "plan": plan,
                "tunables": knobs,
            },
        )["operation_id"]
    assert isinstance(token, int)
    wait_owner(qapp, lambda: controller.operation_id is None)
    assert window.controller is controller
    status = controller.status()
    assert status is not None and status.lifecycle == "done", status
    assert status.workflow == "coherence_fluxdep" and status.committed_seq == 2
    engine = controller.engine
    assert engine is not None
    for record in engine.records("coherence-host"):
        point = record.encoded_record.value
        assert isinstance(point, dict)
        members = point["measurements"]
        assert isinstance(members, list) and len(members) == 1
        echo = members[0]
        assert (
            isinstance(echo, dict)
            and echo["kind"] == "t2echo"
            and echo["status"] == "accepted"
        )
        paths = sim_session.entry.database_path / "workflows" / "coherence-host"
        assert (
            load_run_data(
                paths / record.iteration_dir / "t2echo.h5"
            ).metadata.experiment
            == "twotone/ge/t2echo"
        )
