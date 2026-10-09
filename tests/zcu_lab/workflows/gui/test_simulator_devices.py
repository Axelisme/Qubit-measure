"""Driver-owned fixed limits fail the shipped T1 without clamp or retry."""

import json

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.device.yoko import MAX_CURRENT, YOKOGS200
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.resources.entry import workflow_roots

from ._sim_session import SimSession, wait_owner


class MemoryVisa:
    """VISA transaction collaborator only; all GS200 validation stays real."""

    def __init__(self) -> None:
        self.resource_name = "memory::gs200"
        self.read_termination = ""
        self.write_termination = ""
        self.writes: list[str] = []
        self.closed = False

    def open_resource(self, address: str) -> "MemoryVisa":
        assert address == self.resource_name
        return self

    def query(self, command: str) -> str:
        return {
            "*IDN?": "memory gs200",
            ":OUTPut?": "1",
            ":SOURce:FUNCtion?": "CURR",
            ":SOURce:LEVel?": "0",
        }[command]

    def write(self, command: str) -> None:
        self.writes.append(command)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("sign", [-1, 1])
def test_native_set_device_limit_failure_is_terminal_before_acquisition(
    sim_session: SimSession,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    sign: int,
) -> None:
    visa = MemoryVisa()
    # Replace only VISA: session factory and the GS200 driver remain production.
    monkeypatch.setattr("pyvisa.ResourceManager", lambda: visa)
    app = sim_session.app
    service = app.services.device
    service.start_connect_device(
        ConnectDeviceRequest("YOKOGS200", "limited", visa.resource_name)
    )
    wait_owner(
        qapp,
        lambda: (
            "limited" in service.get_connected_devices()
            and service.active_operation_token("limited") is None
        ),
    )
    assert isinstance(service.get_connected_devices()["limited"], YOKOGS200)
    window = app.window
    window.plan_panel.state.write(
        {("flux_dev",): "limited", ("fluxes", "values"): [sign * MAX_CURRENT * 1.01]},
        window.plan_panel.env,
    )
    run_id = "limit-negative" if sign < 0 else "limit-positive"
    token = window.start(run_id)
    assert isinstance(token, int)
    wait_owner(qapp, lambda: window.controller.operation_id is None)
    status = window.controller.status()
    assert status is not None and status.lifecycle == "failed"
    assert "output limit" in str(status.reason)
    engine = window.controller.engine
    assert engine is not None and engine.records(run_id) == ()
    assert sim_session.call("status", {})["run_id"] == run_id
    assert visa.writes == []  # No clamped value, output mutation or resend.
    roots = workflow_roots(
        result_path=sim_session.entry.result_path,
        database_path=sim_session.entry.database_path,
        run_id=run_id,
    )
    assert not list(roots.database.glob("iter/*/runs/*.h5"))
    events = [
        json.loads(line)
        for line in (roots.result / "journal.jsonl").read_text().splitlines()
    ]
    assert [e["call_seq"] for e in events if e["kind"] == "iteration_started"] == [1]
    assert [e["kind"] for e in events].count("step_failed") == 1
    assert events[-1]["kind"] == "run_ended" and events[-1]["lifecycle"] == "failed"
    assert (
        json.loads((roots.result / "manifest.json").read_text())["lifecycle"]
        == "failed"
    )
