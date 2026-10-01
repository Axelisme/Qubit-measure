"""Tests for RunService — operation-boundary behavior given a RunPermit.

Static preconditions (context readiness, committed-cfg validity, soc capability)
are GuardService's responsibility (see test_guard.py). RunService only handles
the dynamic boundary: tab-busy, lease acquisition, bg submit, and — since
ADR-0066 — the cancel *interpretation* of bg's done/failed (it owns the
stop_event, so it relabels finished vs cancelled here).

Stage 2c: RunService is now an OperationRunner client. Tests use a shared
FakeRunner helper that captures the last spec so callbacks can be driven
directly (replacing the old `_on_run_*` method calls).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.device import FakeDevice, FakeDeviceInfo, GlobalDeviceManager
from zcu_tools.experiment import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer, current_stop_signal
from zcu_tools.experiment.v2_gui.measure.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.measure.adapters.fake.stub import (
    FakeResult,
    FakeRunResult,
)
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    ContextReadiness,
    RunRequest,
)
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.services.guard import GuardService, RunPermit
from zcu_tools.gui.app.measure.services.operation_gate import (
    OperationGate,
    OperationKind,
)
from zcu_tools.gui.app.measure.services.run import RunService
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.resource import CfgRevision
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.event_bus import EventMeta, EventOrigin
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import (
    OperationRunner,
)
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import Module, ProgramV2Cfg

from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._cfg_fakes import make_cfg


def _empty_schema() -> CfgSchema:
    return CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())


def _make_state(
    *,
    readiness: ContextReadiness = ContextReadiness.EMPTY,
) -> tuple[State, str, MagicMock]:
    from zcu_tools.resources.context import MetaDict, ModuleLibrary

    md = MetaDict(None)
    ml = ModuleLibrary(None)
    state = State(
        SessionEnv(
            md=md,
            ml=ml,
            soc=MagicMock(),
            soccfg=MagicMock(),
            readiness=readiness,
        )
    )
    tab_id = "tab-1"
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(requires_soc=True)
    state.add_tab(
        tab_id,
        Session(
            adapter_name="any",
            adapter=adapter,
            cfg=make_cfg(_empty_schema(), state=state),
        ),
    )
    return state, tab_id, adapter


def _plots() -> Plots:
    return Plots(NonPresentingHost())


def _make_permit(state: State, tab_id: str, adapter: MagicMock) -> RunPermit:
    ctx = state.session_env
    return RunPermit(
        tab_id=tab_id,
        adapter_name=state.get_tab(tab_id).adapter_name,
        request=RunRequest(soc=ctx.soc, soccfg=ctx.soccfg, device_snapshot={}),
        accepted_cfg=state.get_tab(tab_id).cfg.accept(
            state.get_tab(tab_id).cfg.observe().ref.revision
        ),
        adapter=adapter,
    )


class _FakeBg:
    """Synchronous background executor stub: calls work() and on_done/on_error inline."""

    def __init__(self, *, fail_submit: bool = False, fail_work: bool = False) -> None:
        self._fail_submit = fail_submit
        self._fail_work = fail_work
        self.last_work: Callable[[], Any] | None = None
        self.last_on_done: Callable[[Any], None] | None = None
        self.last_on_error: Callable[[Exception], None] | None = None

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if self._fail_submit:
            raise RuntimeError("worker boom")
        self.last_work = work
        self.last_on_done = on_done
        self.last_on_error = on_error

    def run_work(self) -> None:
        """Drive the captured work synchronously (for tests that need to inspect outcome)."""
        assert self.last_work is not None
        if self._fail_work:
            assert self.last_on_error is not None
            self.last_on_error(RuntimeError("work boom"))
        else:
            try:
                result = self.last_work()
                assert self.last_on_done is not None
                self.last_on_done(result)
            except Exception as exc:
                assert self.last_on_error is not None
                self.last_on_error(exc)


class _ConstructorFailingProgram:
    @property
    def cfg_model(self) -> ProgramV2Cfg:
        raise NotImplementedError

    def __init__(
        self,
        soccfg: Any,
        cfg: ProgramV2Cfg,
        *,
        modules: list[Any],
        sweep: list[tuple[str, Any]] | None,
    ) -> None:
        raise RuntimeError("builder boom")

    def acquire(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
        raise NotImplementedError


class _NoopModule(Module):
    def __init__(self, name: str) -> None:
        self.name = name

    def init(self, prog: Any) -> None:
        pass

    def run(self, prog: Any, t: Any = 0.0) -> Any:
        return t


def _make_run_service(
    state: State,
    *,
    fail_submit: bool = False,
    mock_emit: bool = True,
) -> tuple[RunService, OperationGate, _FakeBg, OperationHandles]:
    bg = _FakeBg(fail_submit=fail_submit)
    bus = EventBus()
    if mock_emit:
        bus.emit = MagicMock()  # type: ignore[method-assign]
    gate = OperationGate(bus)
    handles = OperationHandles()
    writeback = MagicMock()
    progress = ProgressService(DirectProgressTransport())
    runner = OperationRunner(gate, handles, progress, bg, bus)  # type: ignore[arg-type]
    svc = RunService(state, runner, bus, handles, writeback, gate=gate)
    return svc, gate, bg, handles


@pytest.mark.parametrize(
    ("origin", "expected_kind"),
    [
        (EventOrigin(kind="user"), "user"),
        (EventOrigin(kind="agent", client_id="client-a"), "agent"),
    ],
)
def test_run_started_and_terminal_events_keep_operation_origin(
    origin: EventOrigin, expected_kind: str
) -> None:
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _handles = _make_run_service(state, mock_emit=False)
    observed: list[tuple[object, EventMeta]] = []
    svc._bus.subscribe_with_meta(  # type: ignore[attr-defined]
        RunStartedPayload, lambda payload, meta: observed.append((payload, meta))
    )
    svc._bus.subscribe_with_meta(  # type: ignore[attr-defined]
        RunFinishedPayload, lambda payload, meta: observed.append((payload, meta))
    )

    with svc._bus.origin(origin):  # type: ignore[attr-defined]
        token = svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())
    assert bg.last_on_done is not None
    bg.last_on_done(object())

    assert [type(payload) for payload, _meta in observed] == [
        RunStartedPayload,
        RunFinishedPayload,
    ]
    assert [meta.origin.kind for _payload, meta in observed] == [
        expected_kind,
        expected_kind,
    ]
    assert [meta.origin.operation_id for _payload, meta in observed] == [
        str(token),
        str(token),
    ]
    assert [meta.origin.client_id for _payload, meta in observed] == [
        origin.client_id,
        origin.client_id,
    ]


# ---------------------------------------------------------------------------
# start_run — normal path
# ---------------------------------------------------------------------------


def test_start_run_acquires_lease_and_submits_to_bg():
    state, tab_id, adapter = _make_state()
    svc, gate, bg, _ = _make_run_service(state)

    svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    # bg.submit was called (work captured in _FakeBg)
    assert bg.last_work is not None
    assert gate.has_active(OperationKind.RUN)
    assert state.is_tab_running(tab_id)
    assert not any(
        isinstance(call.args[0], TabInteractionChangedPayload)
        for call in svc._bus.emit.call_args_list  # type: ignore[attr-defined]
    )


def test_worker_executes_permit_after_model_changes_and_releases_lease():
    state, tab_id, adapter = _make_state(readiness=ContextReadiness.ACTIVE)
    state.session_env.md.update(gain=0.25)
    schema = CfgSchema(
        CfgSectionSpec(fields={"gain": ScalarSpec(label="Gain", type=float)}),
        CfgSectionValue(fields={"gain": EvalValue("gain")}),
    )
    cfg = make_cfg(schema, state=state)
    state.get_tab(tab_id).cfg = cfg
    permit = GuardService(state).acquire_run_permit(
        tab_id, expected_revision=CfgRevision(0)
    )
    svc, gate, bg, _ = _make_run_service(state)
    result = object()
    adapter.run.return_value = result
    plots = _plots()

    svc.start_run(permit, plots=plots)
    schema.value.fields["gain"] = DirectValue(0.75)
    state.session_env.md.update(gain=0.75)
    state.version.bump("context")
    cfg.refresh(cfg.observe().ref.revision)
    bg.run_work()

    adapter.run.assert_called_once_with(permit.request, {"gain": 0.25}, plots=plots)
    assert state.get_tab(tab_id).run.result is result
    # Run plots are view-only; the canonical run pane owns only the result.
    assert state.get_tab(tab_id).analysis.plots is None
    assert state.get_tab(tab_id).post_analysis.plots is None
    assert not state.is_tab_running(tab_id)
    assert not gate.has_active(OperationKind.RUN)


def test_artifact_snapshot_keeps_accepted_cfg_after_source_republication() -> None:
    class RecordingSnapshotAdapter(FakeAdapter):
        def run(
            self, req: RunRequest, raw_cfg: dict[str, object], *, plots: Plots
        ) -> FakeRunResult:
            # Controlled execution produces an artifact with the real domain cfg builder.
            return RunRecord(
                cfg=self.build_exp_cfg(raw_cfg, req), result=FakeResult(np.empty(0))
            )

    state, tab_id, _adapter = _make_state(readiness=ContextReadiness.ACTIVE)
    adapter = RecordingSnapshotAdapter()
    state.session_env.md.update(gain=0.25)
    schema = adapter.make_default_cfg(state.session_env)
    schema.value.fields["gain"] = EvalValue("gain")
    cfg = make_cfg(schema, state=state)
    state.get_tab(tab_id).adapter = adapter
    state.get_tab(tab_id).cfg = cfg
    accepted_ref = cfg.observe().ref
    permit = GuardService(state).acquire_run_permit(
        tab_id, expected_revision=accepted_ref.revision
    )
    accepted_basis = permit.accepted_cfg.source_basis
    service, _gate, background, _handles = _make_run_service(state)

    service.start_run(permit, plots=_plots())
    state.session_env.md.update(gain=0.75)
    state.version.bump("context")
    after = cfg.refresh(cfg.observe().ref.revision)
    background.run_work()

    result = state.get_tab(tab_id).run.result
    assert isinstance(result, RunRecord)
    assert result.cfg is not None
    assert result.cfg.gain == 0.25
    assert permit.accepted_cfg.ref == accepted_ref
    assert permit.accepted_cfg.source_basis == accepted_basis
    assert after.ref != accepted_ref
    assert after.source_basis != accepted_basis


def test_start_run_rejects_when_tab_busy():
    state, tab_id, adapter = _make_state()
    state.set_tab_analyzing(tab_id, True)
    svc, gate, bg, _ = _make_run_service(state)

    with pytest.raises(FailedPreconditionError, match="busy") as exc_info:
        svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""
    assert not gate.has_active(OperationKind.RUN)
    assert bg.last_work is None


def test_start_run_releases_lease_when_submit_raises():
    state, tab_id, adapter = _make_state()
    svc, gate, bg, _ = _make_run_service(state, fail_submit=True)

    with pytest.raises(RuntimeError, match="worker boom"):
        svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    assert not gate.has_active(OperationKind.RUN)
    assert not state.is_tab_running(tab_id)
    svc._bus.emit.assert_called_with(  # type: ignore[attr-defined]
        TabInteractionChangedPayload(tab_id, TabInteractionFact.RUN_START_REJECTED)
    )


# ---------------------------------------------------------------------------
# on_terminal — run finished / cancelled / failed paths
# Drive bg.last_on_done / on_error directly to trigger on_terminal.
# ---------------------------------------------------------------------------


def _last_run_finished_payload(bus_emit: MagicMock):
    from zcu_tools.gui.app.measure.events.run import RunFinishedPayload

    for call in reversed(bus_emit.call_args_list):
        (payload,) = call.args
        if isinstance(payload, RunFinishedPayload):
            return payload
    raise AssertionError("no RUN_FINISHED emitted")


def test_run_finished_emits_outcome_finished():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    # Trigger on_done without cancel → finished
    assert bg.last_on_done is not None
    bg.last_on_done(object())

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.tab_id == tab_id
    assert payload.outcome == "finished"
    assert not any(
        isinstance(call.args[0], TabInteractionChangedPayload)
        for call in svc._bus.emit.call_args_list  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize("cancel_requested", [False, True])
def test_result_commit_failure_settles_run_as_failed(
    monkeypatch, cancel_requested: bool
) -> None:
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, handles = _make_run_service(state)
    plots = _plots()
    plots.subplots("trace")
    token = svc.start_run(_make_permit(state, tab_id, adapter), plots=plots)
    monkeypatch.setattr(
        state,
        "update_tab_result",
        MagicMock(side_effect=RuntimeError("commit refused")),
    )
    if cancel_requested:
        assert svc.cancel_run()
    assert bg.last_on_done is not None
    bg.last_on_done(object())

    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "failed"
    assert outcome.error == "commit refused"
    assert svc.active_token is None
    assert not state.is_tab_running(tab_id)
    assert state.get_tab(tab_id).run.result is None


def test_run_failed_emits_outcome_failed_with_message():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("boom"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "failed"
    assert payload.error_message == "boom"


def test_schedule_failure_reports_failed_not_cancelled():
    state, tab_id, adapter = _make_state()

    def run_with_schedule_failure(*_args: Any, **_kwargs: Any) -> object:
        with Schedule(ProgramV2Cfg(), SignalBuffer((1,), dtype=np.float64)) as sched:
            _ = (
                sched.prog_builder(
                    "soc",
                    "soccfg",
                    program_cls=_ConstructorFailingProgram,
                )
                .add(_NoopModule("readout"))
                .build_and_acquire()
            )
        return object()

    adapter.run.side_effect = run_with_schedule_failure
    svc, _gate, bg, handles = _make_run_service(state)
    token = svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    bg.run_work()

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "failed"
    assert payload.error_message == "RuntimeError: builder boom"
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == "failed"
    assert outcome.error == "RuntimeError: builder boom"
    assert state.get_tab(tab_id).run.result is None
    assert not state.is_tab_running(tab_id)


def test_cancel_run_sets_operation_stop_event():
    state, tab_id, adapter = _make_state()
    svc, gate, bg, handles = _make_run_service(state)
    token = svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    assert handles.poll(token) is None  # still pending before cancel
    svc.cancel_run()
    # The stop_event is set, but the operation only settles when the worker
    # self-judges and the terminal handler releases the lease.
    assert gate.has_active(OperationKind.RUN)


def test_cancel_run_stops_experiment_setup_devices_before_first_device():
    state, tab_id, adapter = _make_state()
    dev = FakeDevice(fast_mode=True)
    cfg = FakeDeviceInfo(address="none", output="on", value=1.0, rampstep=0.1)
    GlobalDeviceManager.register_device("run-dev", dev)

    def run_setup_devices(*_args: Any, **_kwargs: Any) -> object:
        exp_cfg = ExpCfgModel(dev={"run-dev": cfg})
        setup_devices(exp_cfg, progress=False)
        return object()

    try:
        adapter.run.side_effect = run_setup_devices
        svc, _gate, bg, handles = _make_run_service(state)
        token = svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())
        svc.cancel_run()
        bg.run_work()

        assert dev.get_output() == "off"
        assert dev.get_value() == 0.0
        outcome = handles.poll(token)
        assert outcome is not None
        assert outcome.status == "cancelled"
    finally:
        GlobalDeviceManager.drop_device("run-dev", ignore_error=True)


def test_run_cancelled_with_partial_result_reports_cancelled_and_keeps_result():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    # Simulate: cancel sets stop_event, then worker returns partial result
    svc.cancel_run()
    partial = object()
    assert bg.last_on_done is not None
    bg.last_on_done(partial)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run.result is partial
    assert not state.is_tab_running(tab_id)


def test_run_cancelled_without_result_reports_cancelled_and_keeps_no_result():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    # Simulate: cancel + worker errors
    svc.cancel_run()
    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("interrupted"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run.result is None
    assert not state.is_tab_running(tab_id)


def test_bg_done_without_cancel_reports_finished():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())
    result = object()

    assert bg.last_on_done is not None
    bg.last_on_done(result)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "finished"
    assert state.get_tab(tab_id).run.result is result


def test_bg_done_after_cancel_reports_cancelled_with_partial():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    svc.cancel_run()  # sets the captured stop_event
    partial = object()
    assert bg.last_on_done is not None
    bg.last_on_done(partial)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run.result is partial


def test_bg_done_after_cancel_and_retry_reset_reports_cancelled():
    state, tab_id, adapter = _make_state()
    partial = object()

    def run_after_retry_reset(*_args: Any, **_kwargs: Any) -> object:
        stop = current_stop_signal()
        assert stop is not None
        stop.clear_stop()
        return partial

    adapter.run.side_effect = run_after_retry_reset
    svc, _gate, bg, handles = _make_run_service(state)
    token = svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    svc.cancel_run()
    bg.run_work()

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run.result is partial
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == "cancelled"


def test_bg_error_without_cancel_reports_failed():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("boom"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "failed"
    assert payload.error_message == "boom"


def test_bg_error_after_cancel_reports_cancelled_without_result():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter), plots=_plots())

    svc.cancel_run()
    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("interrupted"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run.result is None
