"""Window rpc controls contracts through public native/GUI seams."""

import json
from dataclasses import asdict

import pytest
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests.gui.app.workflows._start_fakes import StartHarness

from ._reload_source import InjectedCatalog
from ._synthetic import Context, Plan, Tunables
from ._window_scenarios import install_rpc_pause_workflow
from ._window_support import RpcRequest, form_editable


def test_rpc_pending_status_and_stop(
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, harness, entry = run_attachment
    assert rpc("workflow.status", {})["result"] == {
        "run_id": None,
        "operation_id": None,
        "status": None,
    }
    token = window.start("pending-rpc")
    assert isinstance(token, int)
    expected = {"run_id": "pending-rpc", "operation_id": token, "status": None}
    assert rpc("workflow.status", {})["result"] == expected
    assert rpc("workflow.stop", {"run_id": "pending-rpc"})["result"] == expected
    assert harness.gate.snapshot()
    harness.complete_segment()
    assert not harness.gate.snapshot()
    assert not (entry.result_path / "workflows/pending-rpc").exists()
    assert form_editable(window.plan_panel)


@pytest.mark.parametrize("method", ["workflow.pause", "workflow.resume"])
def test_rpc_pending_commands_do_not_claim_acceptance(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
    method: str,
) -> None:
    window, harness, _entry = attachment
    token = window.start("pending-rpc")
    assert isinstance(token, int)
    native_command = window.pause if method == "workflow.pause" else window.resume
    with pytest.raises(RuntimeError) as native:
        native_command("pending-rpc")
    reply = rpc(method, {"run_id": "pending-rpc"})
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert error["code"] == "controller_error"
    assert error["message"] == str(native.value)
    assert window.controller is not None and window.controller.status() is None
    assert window.controller.operation_id == token and harness.gate.snapshot()


@pytest.mark.parametrize(
    "method", ["workflow.pause", "workflow.resume", "workflow.stop"]
)
@pytest.mark.parametrize(
    "params", [{}, {"run_id": None}, {"run_id": 3}, {"run_id": ""}]
)
def test_rpc_target_parameter_admission(
    rpc: RpcRequest,
    method: str,
    params: dict[str, object],
) -> None:
    # Public route exposes ParamSpec errors; the endpoint envelopes these on the wire.
    with pytest.raises(RemoteError) as raised:
        rpc(method, params)
    assert raised.value.code is ErrorCode.INVALID_PARAMS


@pytest.mark.parametrize(
    "method", ["workflow.pause", "workflow.resume", "workflow.stop"]
)
def test_rpc_without_attached_run_reports_native_target(
    rpc: RpcRequest,
    method: str,
) -> None:
    reply = rpc(method, {"run_id": "absent"})
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert (error["code"], error["reason"], error["data"]) == (
        "precondition_failed",
        "run_mismatch",
        {"expected": "absent", "actual": None},
    )


def test_rpc_pause_resume_retains_token_lease_and_fixed_inputs(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
    catalog_source: InjectedCatalog,
) -> None:
    window, harness, entry = attachment
    attempts, receipts = install_rpc_pause_workflow(
        window, harness, rpc, catalog_source
    )
    token = window.start("rpc-run")
    assert isinstance(token, int)
    harness.complete_segment()
    assert receipts[0]["ok"] is True
    requested = receipts[0]["result"]
    assert isinstance(requested, dict) and isinstance(requested["status"], dict)
    assert (
        requested["operation_id"] == token
        and requested["status"]["lifecycle"] == "pausing"
    )
    paused = rpc("workflow.status", {})["result"]
    assert isinstance(paused, dict) and isinstance(paused["status"], dict)
    assert paused["run_id"] == "rpc-run" and paused["operation_id"] == token
    assert paused["status"]["lifecycle"] == "paused"
    assert window.controller is not None
    native = window.controller.status()
    assert native is not None
    assert paused["status"] == json.loads(json.dumps(asdict(native)))
    assert paused["status"]["progress"][0]["name"] == "rpc-work"
    lease = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    bars = harness.progress.bars_for_operation(token)
    assert lease and bars and harness.handles.live_count() == 1
    assert harness.handles.known_outcome(token) is None

    entry.new_point("later-rpc")
    harness.services.context.use_point(PointUseRequest("later-rpc"))
    harness.services.context.set_alias_rows((AliasRow("qubit", ""),))
    resume_reply = rpc("workflow.resume", {"run_id": "rpc-run"})
    assert resume_reply["ok"] is True
    resumed = resume_reply["result"]
    assert isinstance(resumed, dict)
    assert resumed["run_id"] == "rpc-run"
    assert resumed["operation_id"] == token
    assert not form_editable(window.plan_panel)
    assert (
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()) == lease
    )
    assert harness.progress.bars_for_operation(token) == bars
    assert harness.handles.live_count() == 1
    harness.complete_segment()
    terminal = rpc("workflow.status", {})["result"]
    assert isinstance(terminal, dict) and isinstance(terminal["status"], dict)
    assert (
        terminal["operation_id"] is None
        and terminal["status"]["lifecycle"] == "done"
        and terminal["status"]["run_id"] == "rpc-run"
    )
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert harness.background.error is None
    assert harness.handles.known_outcome(token) is not None
    assert not harness.gate.snapshot() and form_editable(window.plan_panel)


@pytest.mark.parametrize("phase", ["pending", "paused"])
def test_rpc_old_target_does_not_change_current_run(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
    catalog_source: InjectedCatalog,
    phase: str,
) -> None:
    window, harness, _entry = attachment
    attempts: list[str] = []

    def init(_env: InitEnv[Context], _plan: Plan) -> None:
        return None

    @workflow("rpc-target", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append("step")
        window.pause("current")
        yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)
    assert isinstance(window.start("old"), int)
    window.stop("old")
    harness.complete_segment()
    assert isinstance(window.start("current"), int)
    if phase == "paused":
        harness.complete_segment()
    before = rpc("workflow.status", {})["result"]
    gate = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    for method in ("workflow.pause", "workflow.resume", "workflow.stop"):
        reply = rpc(method, {"run_id": "old"})
        error = reply["error"]
        assert reply["ok"] is False and isinstance(error, dict)
        assert (error["code"], error["reason"], error["data"]) == (
            "precondition_failed",
            "run_mismatch",
            {"expected": "old", "actual": "current"},
        )
        assert rpc("workflow.status", {})["result"] == before
        assert (
            tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
            == gate
        )
    assert attempts == ([] if phase == "pending" else ["step"])
    stopped = rpc("workflow.stop", {"run_id": "current"})["result"]
    assert isinstance(stopped, dict)
    if phase == "pending":
        assert stopped["status"] is None
        harness.complete_segment()
    else:
        assert (
            isinstance(stopped["status"], dict)
            and stopped["status"]["lifecycle"] == "stopped"
        )
        assert harness.background.work is None
        rejected = rpc("workflow.resume", {"run_id": "current"})
        error = rejected["error"]
        assert rejected["ok"] is False and isinstance(error, dict)
        assert (error["code"], error["reason"], error["data"]) == (
            "precondition_failed",
            "invalid_run_state",
            {"action": "resume", "lifecycle": "stopped"},
        )
    assert not harness.gate.snapshot() and form_editable(window.plan_panel)


def test_rpc_running_stop_retains_lease_until_owner_delivery(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
    catalog_source: InjectedCatalog,
) -> None:
    window, harness, _entry = attachment
    receipts: list[dict[str, object]] = []

    def init(_env: InitEnv[Context], _plan: Plan) -> None:
        return None

    @workflow("rpc-stop", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        receipts.append(rpc("workflow.stop", {"run_id": "stop-rpc"}))
        assert harness.gate.snapshot()
        yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)
    assert isinstance(window.start("stop-rpc"), int)
    harness.background.execute()
    assert harness.gate.snapshot() and not form_editable(window.plan_panel)
    assert receipts[0]["ok"] is True
    requested = receipts[0]["result"]
    assert isinstance(requested, dict) and isinstance(requested["status"], dict)
    assert requested["status"]["lifecycle"] == "stopping"
    harness.owner.pump_all()
    harness.background.deliver()
    terminal = rpc("workflow.status", {})["result"]
    assert isinstance(terminal, dict) and isinstance(terminal["status"], dict)
    assert (
        terminal["status"]["lifecycle"] == "stopped"
        and terminal["operation_id"] is None
    )
    assert not harness.gate.snapshot() and form_editable(window.plan_panel)
