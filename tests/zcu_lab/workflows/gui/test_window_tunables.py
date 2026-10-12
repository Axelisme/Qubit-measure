"""Window tunables contracts through public native/GUI seams."""

import json
from collections.abc import Mapping

import pytest
from pydantic import JsonValue
from qtpy.QtWidgets import QLabel, QLineEdit, QPushButton
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    InitEnv,
    Next,
    RunMismatch,
    Step,
    TunablesSnapshot,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form import FormEnv, FormErrors, FormWritePath
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests.gui.app.workflows._start_fakes import StartHarness
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.coherence import CoherencePlan, CoherenceTunables

from ._coherence_input import t1_input
from ._reload_source import InjectedCatalog
from ._window_support import RpcRequest


def test_rpc_tunables_running_submits_shared_draft_but_keeps_current_step_capture(
    attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    rpc: RpcRequest,
    catalog_source: InjectedCatalog,
) -> None:
    window, harness, entry = attachment
    observed: list[tuple[float, int]] = []
    replies: list[dict[str, object]] = []

    def init(_env: InitEnv[Calibrated], _plan: CoherencePlan) -> int:
        return 0

    @workflow(
        "tuning-running", plan=CoherencePlan, tunables=CoherenceTunables, init=init
    )
    def step(
        env: WorkflowEnv[Calibrated],
        plan: CoherencePlan,
        tun: CoherenceTunables,
        state: int,
    ) -> Step[int, float]:
        yield from ()
        del env, plan
        if state == 2:
            return Done()
        if state == 0:
            # Public edit installs the accepted panel even before timer delivery.
            replies.append(rpc("workflow.tunables", {"run_id": "running", "batch": []}))
            panel = window.tuning_panel
            assert panel is not None
            panel.panel.state.write({("t1", "guess_us"): {"expr": "10 * 3"}}, FormEnv())
            control = panel.panel.editor.findChild(
                QLineEdit, "form.input.t1.sweep_factor"
            )
            assert isinstance(control, QLineEdit)
            control.setText("-")
            replies.append(
                rpc(
                    "workflow.tunables",
                    {
                        "run_id": "running",
                        "batch": [{"path": ["t1", "reps"], "value": 200}],
                    },
                )
            )
            assert panel.panel.editor.local_parse_errors() is None
            assert t1_input(panel.host.accepted_input.data)["guess_us"] == {
                "expr": "10 * 3"
            }
            assert panel.panel.state.input.data == panel.panel.state.base.data
        observed.append((tun.t1.guess_us, tun.t1.reps))
        return Next(tun.t1.guess_us, state + 1)

    catalog_source.install(step, window)
    assert isinstance(window.start("running"), int)
    harness.complete_segment()
    assert harness.background.error is None
    assert [reply["ok"] for reply in replies] == [True, True]
    for reply, revision in zip(replies, (1, 2), strict=True):
        result = reply["result"]
        assert isinstance(result, dict) and isinstance(result["status"], dict)
        assert (
            result["run_id"] == "running" and result["status"]["revision"] == revision
        )
    assert observed[0][0] == 20.0 and observed[1] == (30.0, 200)
    controller = window.controller
    assert controller is not None and controller.engine is not None
    records = controller.engine.records("running")
    assert [r.revision for r in records] == [0, 2]
    assert [r.encoded_record.value for r in records] == [20.0, 30.0]
    events = [
        json.loads(line)
        for line in (entry.result_path / "workflows/running/journal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    updates = [e for e in events if e["kind"] == "tunables_changed"]
    assert len(updates) == 2
    assert all(e["actor"] == {"kind": "agent", "name": "rpc"} for e in updates)
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("editor", ["gui", "rpc"])
def test_paused_tunables_apply_then_resume_repeats_uncommitted_step(
    attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    editor: str,
) -> None:
    window, harness, entry = attachment
    panel, controller = window.tuning_panel, window.controller
    assert (
        panel is not None and controller is not None and controller.engine is not None
    )
    engine = controller.engine
    token = controller.operation_id
    assert token is not None
    lease = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    assert lease and harness.handles.known_outcome(token) is None
    prior = engine.records("tuning")
    assert len(prior) == 1 and prior[0].revision == 0
    panel.panel.state.write(
        {("t1", "guess_us"): {"expr": "general.ext.guess_us * 2"}}, FormEnv()
    )
    # External Apply resolves the accepted capture, not later live parameters.
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.guess_us", 99.0)
    if editor == "gui":
        applied = panel.apply()
        assert not isinstance(applied, FormErrors) and applied.revision == 1
    else:
        reply = rpc("workflow.tunables", {"run_id": "tuning", "batch": []})
        assert reply["ok"] is True
        result = reply["result"]
        assert isinstance(result, dict) and isinstance(result["status"], dict)
        assert result["status"]["revision"] == 1
    assert engine.records("tuning") == prior
    assert t1_input(panel.host.accepted_input.data)["guess_us"] == {
        "expr": "general.ext.guess_us * 2"
    }
    assert panel.panel.state.base.data == panel.panel.state.input.data
    # Resume also retains the original point/focus.
    entry.new_point("later-tuning")
    context.use_point(PointUseRequest("later-tuning"))
    reply = rpc("workflow.resume", {"run_id": "tuning"})
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict) and result["operation_id"] == token
    assert controller.operation_id == token and harness.handles.live_count() == 1
    assert (
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()) == lease
    )
    harness.complete_segment()
    assert harness.background.error is None
    assert paused_tuning[0][0] == 0
    assert paused_tuning[1][0:2] == (1, 20.0)
    assert paused_tuning[2][0:2] == (1, 40.0)
    assert controller.engine is engine
    records = engine.records("tuning")
    assert records[0] == prior[0] and records[1].revision == 1
    assert records[1].encoded_record.value == 40.0
    status = controller.status()
    assert status is not None and status.lifecycle == "done"
    assert harness.handles.known_outcome(token) is not None
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("value", [-1.0, {"expr": "missing_source * 2"}])
def test_rpc_tunables_invalid_input_retains_draft_without_publishing(
    attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    value: JsonValue,
) -> None:
    del paused_tuning
    window, _harness, _entry = attachment
    panel, controller = window.tuning_panel, window.controller
    assert (
        panel is not None and controller is not None and controller.engine is not None
    )
    before = controller.engine.tunables("tuning")
    accepted = panel.host.accepted_input.data
    reply = rpc(
        "workflow.tunables",
        {
            "run_id": "tuning",
            "batch": [
                {"path": ["t1", "reps"], "value": 200},
                {"path": ["t1", "guess_us"], "value": value},
            ],
        },
    )
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert (error["code"], error["reason"]) == ("invalid_params", "form_errors")
    data = error["data"]
    assert isinstance(data, dict) and isinstance(data["messages"], list)
    assert any(m["path"] == ["t1", "guess_us"] for m in data["messages"])
    assert t1_input(panel.panel.state.input.data)["guess_us"] == value
    assert t1_input(panel.panel.state.input.data)["reps"] == 200
    assert panel.host.accepted_input.data == panel.panel.state.base.data == accepted
    assert controller.engine.tunables("tuning") == before
    panel.discard()
    assert panel.panel.state.input.data == accepted


@pytest.mark.parametrize(
    "batch",
    [
        {},
        [{"path": [], "value": 200}],
        [{"path": ["t1", "reps"], "value": 200, "extra": 1}],
        [
            {"path": ["t1", "reps"], "value": 200},
            {"path": ["t1", "reps"], "value": 300},
        ],
        [{"path": ["calibration", 0], "value": 200}],
        [{"path": ["unknown"], "value": 200}],
        [
            {"path": ["calibration"], "value": {}},
            {"path": ["calibration", "reps"], "value": 200},
        ],
        [{"path": ["calibration", "reps", "0"], "value": 200}],
    ],
)
def test_rpc_tunables_rejects_wire_and_structural_batches_before_draft_write(
    attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    batch: object,
) -> None:
    del paused_tuning
    window, harness, _entry = attachment
    panel, controller = window.tuning_panel, window.controller
    assert (
        panel is not None and controller is not None and controller.engine is not None
    )
    before = (
        panel.panel.state.input.data,
        panel.panel.state.base.data,
        panel.host.accepted_input.data,
        controller.engine.tunables("tuning"),
    )
    token = controller.operation_id
    assert token is not None
    lease = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    assert lease and harness.handles.known_outcome(token) is None
    reply = rpc("workflow.tunables", {"run_id": "tuning", "batch": batch})
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert error["code"] in {"invalid_params", "controller_error"}
    assert (
        panel.panel.state.input.data,
        panel.panel.state.base.data,
        panel.host.accepted_input.data,
        controller.engine.tunables("tuning"),
    ) == before
    assert controller.operation_id == token and harness.handles.live_count() == 1
    assert (
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()) == lease
    )
    assert harness.handles.known_outcome(token) is None


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"run_id": "tuning"},
        {"run_id": "", "batch": []},
        {"run_id": None, "batch": []},
        {"run_id": 3, "batch": []},
    ],
)
def test_rpc_tunables_requires_explicit_target_and_batch(
    rpc: RpcRequest, params: dict[str, object]
) -> None:
    with pytest.raises(RemoteError) as raised:
        rpc("workflow.tunables", params)
    assert raised.value.code is ErrorCode.INVALID_PARAMS


def test_rpc_tunables_absent_and_pending_target_cannot_edit_start_draft(
    attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, harness, entry = attachment
    before = window.tunables_panel.state.input.data
    for actual in (None, "pending"):
        reply = rpc(
            "workflow.tunables",
            {"run_id": "old", "batch": [{"path": ["t1", "guess_us"], "value": 80.0}]},
        )
        error = reply["error"]
        assert reply["ok"] is False and isinstance(error, dict)
        assert (error["reason"], error["data"]) == (
            "run_mismatch",
            {"expected": "old", "actual": actual},
        )
        assert window.tunables_panel.state.input.data == before
        assert window.tuning_panel is None
        if actual is None:
            assert isinstance(window.start("pending"), int)
            before = window.tunables_panel.state.input.data
    with pytest.raises(RuntimeError, match="accepted"):
        window.apply_tunables("pending", {}, actor=Actor("agent", "rpc"))
    reply = rpc("workflow.tunables", {"run_id": "pending", "batch": []})
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert error["code"] == "controller_error" and "accepted" in error["message"]
    assert (
        window.tunables_panel.state.input.data == before and window.tuning_panel is None
    )
    assert not (entry.result_path / "workflows/pending").exists()
    assert harness.gate.snapshot()
    window.stop("pending")
    harness.complete_segment()


def test_rpc_tunables_old_target_preserves_accepted_panel_and_native_run(
    attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
) -> None:
    del paused_tuning
    window, harness, _entry = attachment
    panel, controller = window.tuning_panel, window.controller
    assert (
        panel is not None and controller is not None and controller.engine is not None
    )
    token = controller.operation_id
    assert token is not None
    lease = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
    assert lease and harness.handles.known_outcome(token) is None
    panel.panel.state.write({("t1", "guess_us"): 35.0}, FormEnv())
    before = (
        panel.panel.state.input.data,
        panel.panel.state.base.data,
        panel.host.accepted_input.data,
        controller.engine.tunables("tuning"),
        controller.status(),
        controller.operation_id,
    )
    with pytest.raises(RunMismatch):
        window.apply_tunables("old", {("t1", "reps"): 200}, actor=Actor("agent", "rpc"))
    reply = rpc(
        "workflow.tunables",
        {"run_id": "old", "batch": [{"path": ["t1", "reps"], "value": 200}]},
    )
    error = reply["error"]
    assert reply["ok"] is False and isinstance(error, dict)
    assert error["data"] == {"expected": "old", "actual": "tuning"}
    assert (
        panel.panel.state.input.data,
        panel.panel.state.base.data,
        panel.host.accepted_input.data,
        controller.engine.tunables("tuning"),
        controller.status(),
        controller.operation_id,
    ) == before
    assert controller.operation_id == token
    assert (
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()) == lease
    )
    assert harness.handles.live_count() == 1
    assert harness.handles.known_outcome(token) is None


@pytest.mark.parametrize("failure", ["terminal", "journal"])
def test_rpc_tunables_native_failure_keeps_base_revision_and_local_text(
    attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    failure: str,
) -> None:
    del paused_tuning
    window, harness, entry = attachment
    panel, controller = window.tuning_panel, window.controller
    assert (
        panel is not None and controller is not None and controller.engine is not None
    )
    before = controller.engine.tunables("tuning")
    accepted = panel.host.accepted_input.data
    control = panel.panel.editor.findChild(QLineEdit, "form.input.t1.sweep_factor")
    assert isinstance(control, QLineEdit)
    control.setText("-")
    journal = entry.result_path / "workflows/tuning/journal.jsonl"
    original_journal = journal.read_bytes()
    if failure == "terminal":
        window.stop("tuning")
    else:
        journal.unlink()
        journal.mkdir()
    try:
        reply = rpc(
            "workflow.tunables",
            {
                "run_id": "tuning",
                "batch": [{"path": ["t1", "guess_us"], "value": 45.0}],
            },
        )
        error = reply["error"]
        assert reply["ok"] is False and isinstance(error, dict)
        if failure == "terminal":
            assert (error["code"], error["reason"]) == (
                "precondition_failed",
                "invalid_run_state",
            )
            data = error["data"]
            assert isinstance(data, dict) and data["lifecycle"] == "stopped"
        else:
            assert error["code"] == "controller_error"
            assert "Is a directory" in error["message"]
    finally:
        if failure == "journal":
            # Restore the fault injection even when the request/test fails.
            journal.rmdir()
            journal.write_bytes(original_journal)
    assert controller.engine.tunables("tuning") == before
    assert panel.host.accepted_input.data == panel.panel.state.base.data == accepted
    assert t1_input(panel.panel.state.input.data)["guess_us"] == 45.0
    assert panel.panel.editor.local_parse_errors() is not None
    assert control.text() == "-" and not harness.gate.snapshot()
    assert window.plan_panel.isEnabled() and window.tunables_panel.isEnabled()
    start_button = window.findChild(QPushButton, "workflow.run.start")
    assert start_button is not None and start_button.isEnabled()
    assert not harness.progress.bars_for_owner("workflow")
    # A failed idle update must not leave unfinished presentation behind.
    assert isinstance(window.start("after-failure"), int)
    harness.complete_segment()
    assert harness.background.error is None
    window.close()
    window.close()
    assert not harness.progress.bars_for_owner("workflow")


def test_apply_click_isolates_native_journal_failure_without_accepting_draft(
    run_attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    escaped_qt_errors: list[BaseException],
    settlements: list[int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    del paused_tuning
    window, harness, entry = run_attachment
    controller, panel = window.controller, window.tuning_panel
    assert panel is not None and controller.engine is not None
    token = controller.operation_id
    assert token is not None
    native = controller.engine.tunables("tuning")
    accepted = panel.host.accepted_input.data
    panel.panel.state.write({("t1", "guess_us"): 45.0}, panel.host.env)
    control = panel.panel.editor.findChild(QLineEdit, "form.input.t1.sweep_factor")
    button = panel.findChild(QPushButton, "workflow.tuning.apply")
    message = panel.findChild(QLabel, "workflow.tuning.message")
    assert isinstance(control, QLineEdit) and isinstance(button, QPushButton)
    assert isinstance(message, QLabel)
    control.setText("8.00")
    before_click_text = control.text()
    journal = entry.result_path / "workflows/tuning/journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()
    try:
        button.click()
    finally:
        journal.rmdir()
        journal.write_bytes(original)

    assert escaped_qt_errors == []
    status = controller.status()
    assert status is not None and status.lifecycle == "failed"
    outcome = harness.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "failed"
    assert outcome.error is not None and str(journal) in outcome.error
    assert str(journal) in message.text() and str(journal) in caplog.text
    assert any(
        record.name == "zcu_tools.gui.app.workflows.tuning_panel"
        and record.exc_info is not None
        and isinstance(record.exc_info[1], IsADirectoryError)
        and record.exc_info[1].filename == str(journal)
        for record in caplog.records
    )
    assert controller.engine.tunables("tuning") == native
    assert panel.host.accepted_input.data == panel.panel.state.base.data == accepted
    assert t1_input(panel.panel.state.input.data)["guess_us"] == 45.0
    assert t1_input(panel.panel.state.input.data)["sweep_factor"] == 8.0
    assert control.text() == before_click_text
    assert settlements == [token]
    assert harness.handles.live_count() == 0 and not harness.gate.snapshot()
    assert not harness.progress.bars_for_owner("workflow")
    assert window.plan_panel.isEnabled() and not controller.has_pending_segment
    window.stop("tuning")
    assert window.close() and window.close()
    assert settlements == [token] and escaped_qt_errors == []


def test_window_tunables_journal_failure_reraises_the_original_cause(
    attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del paused_tuning
    window, harness, entry = attachment
    panel = window.tuning_panel
    assert panel is not None
    native_errors: list[Exception] = []
    original_apply = panel.apply_external

    def observe_native_error(
        batch: Mapping[FormWritePath, JsonValue],
        *,
        actor: Actor,
    ) -> TunablesSnapshot | FormErrors:
        try:
            return original_apply(batch, actor=actor)
        except Exception as error:
            native_errors.append(error)
            raise

    # Observe the borrowed public panel boundary without replacing its behavior.
    monkeypatch.setattr(panel, "apply_external", observe_native_error)
    journal = entry.result_path / "workflows/tuning/journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()
    try:
        with pytest.raises(IsADirectoryError) as caught:
            window.apply_tunables(
                "tuning", {("t1", "guess_us"): 45.0}, actor=Actor("agent", "rpc")
            )
    finally:
        journal.rmdir()
        journal.write_bytes(original)
    assert len(native_errors) == 1 and caught.value is native_errors[0]
    assert window.controller is not None
    status = window.controller.status()
    assert status is not None and status.lifecycle == "failed"
    assert not harness.progress.bars_for_owner("workflow")
    window.stop("tuning")
    window.close()
    window.close()
