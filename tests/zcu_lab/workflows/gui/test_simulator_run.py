"""Shipped T1, Qt owner, native simulator and MCP observe one workflow run."""

import json
from dataclasses import asdict
from pathlib import Path
from threading import Event, get_ident

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.workflows import CommittedRecord, TunablesSnapshot
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.session.ports import OperationConflictError
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.resources.entry import workflow_roots

from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.workflows.gui.remote import FormWrite

from ._sim_session import SimSession, wait_owner


def start_native(session: SimSession, inputs: Inputs, starter: str, run_id: str) -> int:
    """Start the shipped declaration via either caller using identical native inputs."""
    plan: list[FormWrite] = [
        {"path": ["flux_dev"], "value": "flux"},
        {"path": ["flux_unit"], "value": "native"},
        {"path": ["fluxes", "kind"], "value": "list"},
        {
            "path": ["fluxes", "values"],
            "value": [
                inputs.flux_value,
                inputs.flux_value + 0.01,
                inputs.flux_value + 0.02,
            ],
        },
        {"path": ["points"], "value": 21},
        {"path": ["pi_length_us"], "value": 0.4},
        {"path": ["qub_pulse", "kind"], "value": "ref"},
        {"path": ["qub_pulse", "name"], "value": "Q1.pulses.pi"},
        {"path": ["readout", "kind"], "value": "ref"},
        {"path": ["readout", "name"], "value": "Q1.readout.ro"},
    ]
    tunables: list[FormWrite] = [
        {"path": ["cal", "span_mhz"], "value": 2.0},
        {"path": ["cal", "rabi_start_factor"], "value": 0.125},
        {"path": ["rounds"], "value": 2},
    ]
    window = session.app.window
    if starter == "gui":
        window.plan_panel.state.write(
            {tuple(e["path"]): e["value"] for e in plan}, window.plan_panel.env
        )
        window.tunables_panel.state.write(
            {tuple(e["path"]): e["value"] for e in tunables}, FormEnv()
        )
        token = window.start(run_id)
    else:
        token = session.call(
            "start", {"run_id": run_id, "plan": plan, "tunables": tunables}
        )["operation_id"]
    assert isinstance(token, int)
    return token


def assert_paused_ownership(
    session: SimSession, run_id: str, token: int
) -> tuple[CommittedRecord, ...]:
    """Check retained lease, committed history and stale-command isolation."""
    window, service = session.app.window, session.app.services.device
    controller = window.controller
    engine = controller.engine
    assert engine is not None
    prior = engine.records(run_id)
    assert len(prior) == 1 and prior[0].revision == 0
    assert controller.operation_id == token and not window.plan_panel.isEnabled()
    with pytest.raises(OperationConflictError):
        service.start_connect_device(
            ConnectDeviceRequest("FakeDevice", "blocked", "none")
        )
    before = controller.status()
    for method in ("pause", "resume", "stop"):
        with pytest.raises(RuntimeError, match="run_mismatch"):
            session.call(method, {"run_id": "stale"})
    assert controller.status() == before
    return prior


def apply_and_resume(
    session: SimSession,
    starter: str,
    run_id: str,
    token: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply revision one and resume the same handle without reading devices."""
    window = session.app.window
    controller, panel = window.controller, window.tuning_panel
    engine = controller.engine
    assert panel is not None and engine is not None
    prior = engine.records(run_id)
    if starter == "gui":
        session.call(
            "tunables", {"run_id": run_id, "batch": [{"path": ["rounds"], "value": 3}]}
        )
    else:
        panel.panel.state.write({("rounds",): 3}, panel.host.env)
        applied = panel.apply()
        assert isinstance(applied, TunablesSnapshot) and applied.revision == 1
    assert engine.records(run_id) == prior
    devices = session.app.services.device.get_connected_devices()
    monitor = devices["monitor"]
    assert isinstance(monitor, FakeDevice)
    monitor.set_value(0.37)

    assert controller.operation_id == token and engine.records(run_id) == prior
    owner_thread = get_ident()
    reads: list[tuple[str, int]] = []
    with monkeypatch.context() as patch:
        for name, driver in devices.items():
            original = driver.get_info

            def read(original=original, name=name):
                reads.append((name, get_ident()))
                return original()

            patch.setattr(driver, "get_info", read)
        if starter == "gui":
            assert window.resume(run_id) == token
        else:
            assert session.call("resume", {"run_id": run_id})["operation_id"] == token
    # Submitted worker acquisitions may already be capturing; Resume itself never reads.
    assert all(thread != owner_thread for _, thread in reads)
    assert controller.operation_id == token and engine.records(run_id) == prior


def assert_native_files(
    session: SimSession, data_root: Path, records: tuple[CommittedRecord, ...]
) -> None:
    """Load the native acquisitions, verifying captured metadata and measured T1."""
    for record in records:
        assert record.run_files[-1].endswith("-run_t1.h5")
        native, metadata = load_run_with_metadata(
            data_root / record.iteration_dir / record.run_files[-1],
            spec=T1Exp().persistence_spec(),
        )
        assert native.cfg is not None and native.cfg.rounds == record.revision + 2
        assert str(metadata.snapshot.entry_id) == session.entry.entry_id
        assert metadata.snapshot.point == "source" and metadata.completion == "complete"
        assert metadata.device_snapshot is not None
        assert metadata.device_snapshot["monitor"]["value"] == (
            0.37 if record.revision else 0.0
        )
        assert metadata.device_snapshot["flux"]["value_unit"] == "native"
        assert native.result.times.shape == native.result.signals.shape == (21,)
        assert np.isfinite(native.result.signals).all()
        point = record.encoded_record.value
        assert isinstance(point, dict) and point["reason"] is None
        assert point["lifetime_us"] == pytest.approx(20.0, rel=0.2)


def assert_persisted_records(
    session: SimSession, run_id: str, prior: tuple[CommittedRecord, ...]
) -> None:
    """Compare GUI/engine/MCP and disk authority after terminal Stop."""
    controller = session.app.window.controller
    status, engine = controller.status(), controller.engine
    assert status is not None and status.lifecycle == "stopped", status
    assert engine is not None
    assert status.run_id == run_id and status.committed_seq == 2
    records = engine.records(run_id)
    assert records[0] == prior[0] and [r.revision for r in records] == [0, 1]
    assert records[1].call_seq > records[0].call_seq + 1
    assert session.call("status", {}) == {
        "run_id": run_id,
        "operation_id": None,
        "status": json.loads(json.dumps(asdict(status))),
    }
    expected = [
        {
            "commit_seq": r.commit_seq,
            "call_seq": r.call_seq,
            "revision": r.revision,
            "encoded_record": asdict(r.encoded_record),
            "iteration_dir": r.iteration_dir,
            "run_files": list(r.run_files),
        }
        for r in records
    ]
    assert session.call("records", {"run_id": run_id}) == {
        "run_id": run_id,
        "records": expected,
    }
    assert (
        session.call("records", {"run_id": run_id, "since": 1})["records"]
        == expected[1:]
    )
    roots = workflow_roots(
        result_path=session.entry.result_path,
        database_path=session.entry.database_path,
        run_id=run_id,
    )
    assert roots.result == session.entry.result_path / "workflows" / run_id
    assert roots.database == session.entry.database_path / "workflows" / run_id
    assert (
        json.loads((roots.result / "manifest.json").read_text())["lifecycle"]
        == "stopped"
    )
    events = [
        json.loads(line)
        for line in (roots.result / "journal.jsonl").read_text().splitlines()
    ]
    resumed = next(e for e in events if e["kind"] == "resumed")
    assert resumed["commit_seq"] == 1 and resumed["revision"] == 1
    assert any(e["kind"] == "step_discarded" and e["reason"] == "pause" for e in events)
    commits = [e for e in events if e["kind"] == "step_committed"]
    assert [e["encoded_record"]["value"] for e in commits] == [
        r.encoded_record.value for r in records
    ]
    assert_native_files(session, roots.database, records)


def hold_control_setpoints(
    device: FakeDevice,
    pause_value: float,
    stop_value: float,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Event, Event]:
    """Install one-shot setup holds and return Pause/Stop entry notifications.

    The first setup at each selected value waits for its native cancellation
    event before forwarding to the real driver. Repeated setups run normally.
    Missing cancellation events or absent owner requests fail within five seconds.
    Monkeypatch restores the public setup method when the test finishes.
    """
    pause_entered, stop_entered = Event(), Event()
    gates = {pause_value: pause_entered, stop_value: stop_entered}
    original_setup = device.setup

    def setup_at_controlled_setpoint(
        cfg: FakeDeviceInfo,
        *,
        progress: bool = True,
        stop_event: Event | None = None,
    ) -> None:
        """Hold once at a selected value, then forward unchanged setup arguments."""
        gate = gates.get(cfg.value)
        if gate is not None and not gate.is_set():
            assert stop_event is not None
            gate.set()
            assert stop_event.wait(5), "owner did not cancel the in-flight setup"
        original_setup(cfg, progress=progress, stop_event=stop_event)

    monkeypatch.setattr(device, "setup", setup_at_controlled_setpoint)
    return pause_entered, stop_entered


@pytest.mark.parametrize("starter", ["gui", "mcp"])
def test_native_pause_apply_resume_stop_retains_records_and_exact_entry_roots(
    sim_session: SimSession,
    inputs: Inputs,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    starter: str,
) -> None:
    session, app = sim_session, sim_session.app
    window, controller = app.window, app.window.controller
    service = app.services.device
    service.start_connect_device(ConnectDeviceRequest("FakeDevice", "monitor", "none"))
    wait_owner(
        qapp,
        lambda: (
            "monitor" in service.get_connected_devices()
            and service.active_operation_token("monitor") is None
        ),
    )
    flux = service.get_connected_devices()["flux"]
    assert isinstance(flux, FakeDevice)
    pause_entered, stop_entered = hold_control_setpoints(
        flux, inputs.flux_value + 0.01, inputs.flux_value + 0.02, monkeypatch
    )
    run_id = "native-" + starter
    token = start_native(session, inputs, starter, run_id)
    wait_owner(qapp, pause_entered.is_set)
    status = controller.status()
    assert status is not None and status.lifecycle == "running"
    assert status.call_seq == 2 and status.committed_seq == 1
    # A commit alone permits a boundary Pause with no discarded call. Enter the
    # next device effect first so the strict call-sequence gap proves discard.
    if starter == "gui":
        session.call("pause", {"run_id": run_id})
    else:
        window.pause(run_id)
    wait_owner(
        qapp,
        lambda: (
            (status := controller.status()) is not None and status.lifecycle == "paused"
        ),
    )
    wait_owner(qapp, lambda: not controller.has_pending_segment)
    prior = assert_paused_ownership(session, run_id, token)
    apply_and_resume(session, starter, run_id, token, monkeypatch)
    # Hold the final setpoint too: owner delivery cannot race all the way to Done
    # while the test is observing the second committed record before Stop.
    wait_owner(qapp, stop_entered.is_set)
    status = controller.status()
    assert status is not None and status.lifecycle == "running"
    assert status.call_seq == 4 and status.committed_seq == 2
    if starter == "gui":
        window.stop(run_id)
    else:
        session.call("stop", {"run_id": run_id})
    wait_owner(qapp, lambda: controller.operation_id is None)
    assert_persisted_records(session, run_id, prior)
    names = window.figure_names(run_id)
    assert session.call("figures", {"run_id": run_id}) == {
        "run_id": run_id,
        "names": list(names),
    }
    image = session.image(run_id, "t1/decay")
    assert image.data == {"run_id": run_id, "name": "t1/decay"}
    assert image.images[0].data == window.figure_png(run_id, "t1/decay")
    assert image.images[0].data.startswith(b"\x89PNG\r\n\x1a\n")
    # Terminal releases admission; the same session can now connect again.
    service.start_connect_device(ConnectDeviceRequest("FakeDevice", "after", "none"))
    wait_owner(
        qapp,
        lambda: (
            "after" in service.get_connected_devices()
            and service.active_operation_token("after") is None
        ),
    )
