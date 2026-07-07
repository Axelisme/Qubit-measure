"""Tests for RunService — operation-boundary behavior given a RunPermit.

Static preconditions (context readiness, committed-cfg validity, soc capability)
are GuardService's responsibility (see test_guard.py). RunService only handles
the dynamic boundary: tab-busy, lease acquisition, bg submit, and — since
ADR-0019 — the cancel *interpretation* of bg's done/failed (it owns the
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
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer, current_stop_signal
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    RunRequest,
)
from zcu_tools.gui.app.main.services.guard import RunPermit
from zcu_tools.gui.app.main.services.operation_gate import OperationGate, OperationKind
from zcu_tools.gui.app.main.services.run import RunService
from zcu_tools.gui.app.main.state import ExpContext, Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import (
    OperationRunner,
)
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.program.v2 import Module, ProgramV2Cfg

from ._progress_fakes import DirectProgressTransport


def _empty_schema() -> CfgSchema:
    return CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())


def _make_state() -> tuple[State, str, MagicMock]:
    md = MagicMock()
    ml = MagicMock()
    state = State(ExpContext(md=md, ml=ml, soc=MagicMock(), soccfg=MagicMock()))
    tab_id = "tab-1"
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(requires_soc=True)
    state.add_tab(
        tab_id,
        Session(adapter_name="any", adapter=adapter, cfg_schema=_empty_schema()),
    )
    return state, tab_id, adapter


def _make_permit(state: State, tab_id: str, adapter: MagicMock) -> RunPermit:
    ctx = state.exp_context
    return RunPermit(
        tab_id=tab_id,
        request=RunRequest(md=ctx.md, ml=ctx.ml, soc=ctx.soc, soccfg=ctx.soccfg),
        schema=state.get_tab(tab_id).cfg_schema,
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
) -> tuple[RunService, OperationGate, _FakeBg, OperationHandles]:
    bg = _FakeBg(fail_submit=fail_submit)
    bus = EventBus()
    bus.emit = MagicMock()  # type: ignore[method-assign]
    gate = OperationGate()
    handles = OperationHandles()
    writeback = MagicMock()
    progress = ProgressService(DirectProgressTransport())
    runner = OperationRunner(gate, handles, progress, bg)  # type: ignore[arg-type]
    svc = RunService(state, runner, bus, handles, writeback)
    return svc, gate, bg, handles


# ---------------------------------------------------------------------------
# start_run — normal path
# ---------------------------------------------------------------------------


def test_start_run_acquires_lease_and_submits_to_bg():
    state, tab_id, adapter = _make_state()
    svc, gate, bg, _ = _make_run_service(state)

    svc.start_run(_make_permit(state, tab_id, adapter))

    # bg.submit was called (work captured in _FakeBg)
    assert bg.last_work is not None
    assert gate.has_active(OperationKind.RUN)
    assert state.is_tab_running(tab_id)


def test_start_run_rejects_when_tab_busy():
    state, tab_id, adapter = _make_state()
    state.set_tab_analyzing(tab_id, True)
    svc, gate, bg, _ = _make_run_service(state)

    with pytest.raises(RuntimeError, match="busy"):
        svc.start_run(_make_permit(state, tab_id, adapter))

    assert not gate.has_active(OperationKind.RUN)
    assert bg.last_work is None


def test_start_run_releases_lease_when_submit_raises():
    state, tab_id, adapter = _make_state()
    svc, gate, bg, _ = _make_run_service(state, fail_submit=True)

    with pytest.raises(RuntimeError, match="worker boom"):
        svc.start_run(_make_permit(state, tab_id, adapter))

    assert not gate.has_active(OperationKind.RUN)
    assert not state.is_tab_running(tab_id)


# ---------------------------------------------------------------------------
# on_terminal — run finished / cancelled / failed paths
# Drive bg.last_on_done / on_error directly to trigger on_terminal.
# ---------------------------------------------------------------------------


def _last_run_finished_payload(bus_emit: MagicMock):
    from zcu_tools.gui.app.main.events.run import RunFinishedPayload

    for call in reversed(bus_emit.call_args_list):
        (payload,) = call.args
        if isinstance(payload, RunFinishedPayload):
            return payload
    raise AssertionError("no RUN_FINISHED emitted")


def test_run_finished_emits_outcome_finished():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    # Trigger on_done without cancel → finished
    assert bg.last_on_done is not None
    bg.last_on_done(object())

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.tab_id == tab_id
    assert payload.outcome == "finished"


def test_run_failed_emits_outcome_failed_with_message():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("boom"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "failed"
    assert payload.error_message == "boom"


def test_schedule_failure_reports_failed_not_cancelled():
    state, tab_id, adapter = _make_state()

    def run_with_schedule_failure(*_args: Any) -> object:
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
    token = svc.start_run(_make_permit(state, tab_id, adapter))

    bg.run_work()

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "failed"
    assert payload.error_message == "RuntimeError: builder boom"
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == "failed"
    assert outcome.error == "RuntimeError: builder boom"
    assert state.get_tab(tab_id).run_result is None
    assert not state.is_tab_running(tab_id)


def test_cancel_run_sets_operation_stop_event():
    state, tab_id, adapter = _make_state()
    svc, gate, bg, handles = _make_run_service(state)
    token = svc.start_run(_make_permit(state, tab_id, adapter))

    assert handles.poll(token) is None  # still pending before cancel
    svc.cancel_run()
    # The stop_event is set, but the operation only settles when the worker
    # self-judges and the terminal handler releases the lease.
    assert gate.has_active(OperationKind.RUN)


def test_run_cancelled_with_partial_result_reports_cancelled_and_keeps_result():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    # Simulate: cancel sets stop_event, then worker returns partial result
    svc.cancel_run()
    partial = object()
    assert bg.last_on_done is not None
    bg.last_on_done(partial)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run_result is partial
    assert not state.is_tab_running(tab_id)


def test_run_cancelled_without_result_reports_cancelled_and_keeps_no_result():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    # Simulate: cancel + worker errors
    svc.cancel_run()
    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("interrupted"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run_result is None
    assert not state.is_tab_running(tab_id)


def test_bg_done_without_cancel_reports_finished():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))
    result = object()

    assert bg.last_on_done is not None
    bg.last_on_done(result)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "finished"
    assert state.get_tab(tab_id).run_result is result


def test_bg_done_after_cancel_reports_cancelled_with_partial():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    svc.cancel_run()  # sets the captured stop_event
    partial = object()
    assert bg.last_on_done is not None
    bg.last_on_done(partial)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run_result is partial


def test_bg_done_after_cancel_and_retry_reset_reports_cancelled():
    state, tab_id, adapter = _make_state()
    partial = object()

    def run_after_retry_reset(*_args: Any) -> object:
        stop = current_stop_signal()
        assert stop is not None
        stop.clear_stop()
        return partial

    adapter.run.side_effect = run_after_retry_reset
    svc, _gate, bg, handles = _make_run_service(state)
    token = svc.start_run(_make_permit(state, tab_id, adapter))

    svc.cancel_run()
    bg.run_work()

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run_result is partial
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == "cancelled"


def test_bg_error_without_cancel_reports_failed():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("boom"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "failed"
    assert payload.error_message == "boom"


def test_bg_error_after_cancel_reports_cancelled_without_result():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg, _ = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    svc.cancel_run()
    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("interrupted"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run_result is None
