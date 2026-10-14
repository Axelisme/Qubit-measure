"""Tests for RunService — operation-boundary behavior given a RunInputs.

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

from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.cfg_model import ConfigBase, Expr, Range
from zcu_tools.device import FakeDevice
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils.device import set_flux
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.gui.app.measure.adapter import (
    ContextReadiness,
    RunRequest,
)
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.services.operation_gate import (
    OperationKind,
)
from zcu_tools.gui.event_bus import EventMeta, EventOrigin
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.program.v2 import ProgramV2Cfg

from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.app.measure._form_fakes import make_form
from tests.gui.app.measure._synthetic import (
    SampleAdapter,
    SampleCfg,
    SampleResult,
    SampleRun,
)

from ._run_support import (
    ConstructorFailingProgram,
    NoopModule,
    last_run_finished_payload,
    make_inputs,
    make_plots,
    make_run_service,
    make_state,
)


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
    state, tab_id, adapter = make_state()
    svc, _gate, bg, _handles = make_run_service(state)
    observed: list[tuple[object, EventMeta]] = []
    bg.bus.subscribe_with_meta(
        RunStartedPayload, lambda payload, meta: observed.append((payload, meta))
    )
    bg.bus.subscribe_with_meta(
        RunFinishedPayload, lambda payload, meta: observed.append((payload, meta))
    )

    with bg.bus.origin(origin):
        token = svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())
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


@pytest.mark.parametrize("context_free", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_run_publishes_admitted_source_despite_later_global_change(
    result_sources: tuple[ParameterContext, ParameterContext],
    context_free: bool,
    cancelled: bool,
) -> None:
    state, tab_id, adapter = make_state(readiness=ContextReadiness.ACTIVE)
    q1, q2 = result_sources
    captured = None if context_free else q1
    tab = state.get_tab(tab_id)
    state.update_tab_result(tab_id, object(), parameters=q2, source_operation_id=91)
    state.update_tab_analyze(tab_id, object(), None)
    state.set_context(replace(state.session_env, parameters=captured))
    inputs = make_inputs(state, tab_id, adapter)
    assert inputs.parameters is captured
    service, gate, background, _handles = make_run_service(state)
    result = object()
    adapter.run.return_value = result
    token = service.start_run(inputs, plots=make_plots())

    assert tab.run.result is None
    assert tab.run.parameters is None
    assert tab.analysis.result is None
    state.set_context(replace(state.session_env, parameters=q2))
    if cancelled:
        assert service.cancel_run() is True
    background.run_work()

    assert tab.run.result is result
    assert tab.run.source_operation_id == token
    assert tab.run.parameters is captured
    if tab.run.parameters is not None:
        assert tab.run.parameters.resolve_path("source_qubit.rate") == "Q1.rate"
        assert tab.run.parameters.value("source_qubit.rate") == 1.0
    assert not state.is_tab_running(tab_id)
    assert not gate.has_active(OperationKind.RUN)

    next_inputs = make_inputs(state, tab_id, adapter)
    next_token = service.start_run(next_inputs, plots=make_plots())
    assert tab.run.parameters is None
    assert tab.run.source_operation_id is None
    background.run_work()
    next_source = state.get_tab(tab_id).run.parameters
    assert next_source is not None
    assert next_source is q2
    assert next_source.value("source_qubit.rate") == 2.0
    assert tab.run.source_operation_id == next_token

    state.clear_tab_results(tab_id)
    assert tab.run.result is None
    assert tab.run.parameters is None
    assert tab.run.source_operation_id is None


def test_cancelled_run_without_data_keeps_no_previous_source(
    result_sources: tuple[ParameterContext, ParameterContext],
) -> None:
    state, tab_id, adapter = make_state(readiness=ContextReadiness.ACTIVE)
    q1, q2 = result_sources
    tab = state.get_tab(tab_id)
    state.update_tab_result(tab_id, object(), parameters=q1, source_operation_id=91)
    state.set_context(replace(state.session_env, parameters=q2))
    inputs = make_inputs(state, tab_id, adapter)
    service, _gate, background, _handles = make_run_service(state)
    service.start_run(inputs, plots=make_plots())
    service.cancel_run()
    assert background.last_on_error is not None
    background.last_on_error(RuntimeError("interrupted"))

    assert tab.run.result is None
    assert tab.run.parameters is None
    assert tab.run.source_operation_id is None
    assert not state.is_tab_running(tab_id)


def test_start_run_acquires_lease_and_submits_to_bg():
    state, tab_id, adapter = make_state()
    svc, gate, bg, _ = make_run_service(state)

    svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    # bg.submit was called (work captured in FakeBg)
    assert bg.last_work is not None
    assert gate.has_active(OperationKind.RUN)
    assert state.is_tab_running(tab_id)
    assert not any(
        isinstance(payload, TabInteractionChangedPayload) for payload in bg.events
    )


def test_worker_executes_permit_after_model_changes_and_releases_lease():
    state, tab_id, adapter = make_state(readiness=ContextReadiness.ACTIVE)
    state.set_context(
        replace(state.session_env, parameters=parameter_capture(gain=0.25))
    )

    class CapturedGain(ConfigBase):
        gain: float

    cfg = make_form(Form(CapturedGain, {"gain": Expr("general.ext.gain")}), state=state)
    state.get_tab(tab_id).cfg = cfg
    inputs = make_inputs(state, tab_id, adapter)
    svc, gate, bg, _ = make_run_service(state)
    result = object()
    adapter.run.return_value = result
    plots = make_plots()

    svc.start_run(inputs, plots=plots)
    state.set_context(
        replace(state.session_env, parameters=parameter_capture(gain=0.75))
    )
    state.version.bump("context")
    assert cfg.accept().values["gain"] == 0.25
    bg.run_work()

    context = adapter.run.call_args.kwargs["context"]
    adapter.run.assert_called_once_with(inputs.request, {"gain": 0.25}, context=context)
    assert context.plots is plots
    assert context.soc is inputs.request.soc
    assert context.soccfg is inputs.request.soccfg
    assert state.get_tab(tab_id).run.result is result
    # Run plots are view-only; the canonical run pane owns only the result.
    assert state.get_tab(tab_id).analysis.plots is None
    assert state.get_tab(tab_id).post_analysis.plots is None
    assert not state.is_tab_running(tab_id)
    assert not gate.has_active(OperationKind.RUN)


def test_artifact_snapshot_keeps_accepted_cfg_after_source_republication() -> None:
    class RecordingSnapshotAdapter(SampleAdapter):
        @staticmethod
        def make_form(ctx: FormContext) -> Form[SampleCfg]:
            del ctx
            return Form(
                SampleCfg,
                {
                    "gain": Expr("general.ext.gain"),
                    "sweep": Range.between(5, 6, expts=11),
                },
            )

        def run(
            self,
            req: RunRequest,
            raw_cfg: Mapping[str, ExperimentCfgValue],
            *,
            context: RunContext,
        ) -> SampleRun:
            # Controlled execution produces an artifact with the real domain cfg builder.
            return RunRecord(
                cfg=self.build_exp_cfg(dict(raw_cfg), req),
                result=SampleResult(np.empty(0)),
            )

    state, tab_id, _adapter = make_state(readiness=ContextReadiness.ACTIVE)
    adapter = RecordingSnapshotAdapter()
    state.set_context(
        replace(state.session_env, parameters=parameter_capture(gain=0.25))
    )
    cfg = make_form(adapter.form, state=state)
    state.get_tab(tab_id).adapter = adapter
    state.get_tab(tab_id).cfg = cfg
    inputs = make_inputs(state, tab_id, adapter)
    accepted_basis = inputs.accepted_cfg.source_basis
    service, _gate, background, _handles = make_run_service(state)

    service.start_run(inputs, plots=make_plots())
    state.set_context(
        replace(state.session_env, parameters=parameter_capture(gain=0.75))
    )
    state.version.bump("context")
    assert cfg.accept().source_basis == accepted_basis
    background.run_work()

    result = state.get_tab(tab_id).run.result
    assert isinstance(result, RunRecord)
    assert result.cfg is not None
    assert result.cfg.gain == 0.25
    assert inputs.accepted_cfg.source_basis == accepted_basis
    cfg.reset()
    assert cfg.accept().source_basis != accepted_basis
    assert cfg.accept().values["gain"] == 0.75


def test_start_run_rejects_when_tab_busy():
    state, tab_id, adapter = make_state()
    state.set_tab_analyzing(tab_id, analyzing=True)
    svc, gate, bg, _ = make_run_service(state)

    with pytest.raises(FailedPreconditionError, match="busy") as exc_info:
        svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""
    assert not gate.has_active(OperationKind.RUN)
    assert bg.last_work is None


def test_start_run_releases_lease_when_submit_raises():
    state, tab_id, adapter = make_state()
    svc, gate, bg, _ = make_run_service(state, fail_submit=True)

    with pytest.raises(RuntimeError, match="worker boom"):
        svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    assert not gate.has_active(OperationKind.RUN)
    assert not state.is_tab_running(tab_id)
    assert bg.events[-1] == TabInteractionChangedPayload(
        tab_id, TabInteractionFact.RUN_START_REJECTED
    )


def test_run_finished_emits_outcome_finished():
    state, tab_id, adapter = make_state()
    svc, _gate, bg, _ = make_run_service(state)
    svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    # Trigger on_done without cancel → finished
    assert bg.last_on_done is not None
    bg.last_on_done(object())

    payload = last_run_finished_payload(bg.events)
    assert payload.tab_id == tab_id
    assert payload.outcome == "finished"
    assert not any(
        isinstance(payload, TabInteractionChangedPayload) for payload in bg.events
    )


@pytest.mark.parametrize("cancel_requested", [False, True])
def test_result_commit_failure_settles_run_as_failed(
    monkeypatch, cancel_requested: bool
) -> None:
    state, tab_id, adapter = make_state()
    svc, _gate, bg, handles = make_run_service(state)
    plots = make_plots()
    plots.subplots("trace")
    token = svc.start_run(make_inputs(state, tab_id, adapter), plots=plots)
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
    state, tab_id, adapter = make_state()
    svc, _gate, bg, _ = make_run_service(state)
    svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("boom"))

    payload = last_run_finished_payload(bg.events)
    assert payload.outcome == "failed"
    assert payload.error_message == "boom"


def test_schedule_failure_reports_failed_not_cancelled():
    state, tab_id, adapter = make_state()

    def run_with_schedule_failure(*_args: Any, context: RunContext) -> object:
        with Schedule(
            ProgramV2Cfg(),
            SignalBuffer((1,), dtype=np.float64),
            stop=context.cancel_signal,
        ) as sched:
            _ = (
                sched.prog_builder(
                    "soc",
                    "soccfg",
                    program_cls=ConstructorFailingProgram,
                )
                .add(NoopModule("readout"))
                .build_and_acquire()
            )
        return object()

    adapter.run.side_effect = run_with_schedule_failure
    svc, _gate, bg, handles = make_run_service(state)
    token = svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    bg.run_work()

    payload = last_run_finished_payload(bg.events)
    assert payload.outcome == "failed"
    assert payload.error_message == "RuntimeError: builder boom"
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == "failed"
    assert outcome.error == "RuntimeError: builder boom"
    assert state.get_tab(tab_id).run.result is None
    assert not state.is_tab_running(tab_id)


def test_cancel_run_sets_operation_stop_event():
    state, tab_id, adapter = make_state()
    svc, gate, _bg, handles = make_run_service(state)
    token = svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    assert handles.poll(token) is None  # still pending before cancel
    svc.cancel_run()
    # The stop_event is set, but the operation only settles when the worker
    # self-judges and the terminal handler releases the lease.
    assert gate.has_active(OperationKind.RUN)


def test_cancel_run_stops_selected_flux_before_first_write():
    state, tab_id, adapter = make_state()
    dev = FakeDevice(fast_mode=True)

    def run_flux(*_args: Any, context: RunContext) -> object:
        set_flux(
            context.devices["run-dev"], 1.0, unit="native", stop=context.cancel_signal
        )
        return object()

    adapter.run.side_effect = run_flux
    svc, _gate, bg, handles = make_run_service(state, devices={"run-dev": dev})
    token = svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())
    svc.cancel_run()
    bg.run_work()

    assert dev.get_output() == "off"
    assert dev.get_value() == 0.0
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == "cancelled"


def test_each_run_freezes_driver_mapping_and_owns_fresh_cancellation() -> None:
    state, tab_id, adapter = make_state()
    first_device = FakeDevice(fast_mode=True)
    second_device = FakeDevice(fast_mode=True)
    devices = {"flux": first_device}
    contexts: list[RunContext] = []

    def run_flux(*_args: Any, context: RunContext) -> object:
        contexts.append(context)
        set_flux(
            context.devices["flux"], 0.5, unit="native", stop=context.cancel_signal
        )
        if len(contexts) == 1:
            context.cancel_signal.set_error("failed", "first run failed", None)
        return object()

    adapter.run.side_effect = run_flux
    service, gate, background, handles = make_run_service(state, devices=devices)
    inputs = make_inputs(state, tab_id, adapter)
    first_token = service.start_run(inputs, plots=make_plots())
    devices["flux"] = second_device
    background.run_work()

    first_outcome = handles.poll(first_token)
    assert first_outcome is not None and first_outcome.status == "failed"
    assert first_device.get_value() == 0.5
    assert second_device.get_value() == 0.0
    assert not gate.has_active(OperationKind.RUN)

    second_token = service.start_run(inputs, plots=make_plots())
    background.run_work()

    second_outcome = handles.poll(second_token)
    assert second_outcome is not None and second_outcome.status == "finished"
    assert second_device.get_value() == 0.5
    assert contexts[0].cancel_signal is not contexts[1].cancel_signal
    assert contexts[1].cancel_signal.error is None
    assert contexts[0].plots is not contexts[1].plots
    assert not gate.has_active(OperationKind.RUN)


def test_run_cancelled_with_partial_result_reports_cancelled_and_keeps_result():
    state, tab_id, adapter = make_state()
    svc, _gate, bg, _ = make_run_service(state)
    svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    # Simulate: cancel sets stop_event, then worker returns partial result
    svc.cancel_run()
    partial = object()
    assert bg.last_on_done is not None
    bg.last_on_done(partial)

    payload = last_run_finished_payload(bg.events)
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run.result is partial
    assert not state.is_tab_running(tab_id)


def test_run_cancelled_without_result_reports_cancelled_and_keeps_no_result():
    state, tab_id, adapter = make_state()
    svc, _gate, bg, _ = make_run_service(state)
    svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    # Simulate: cancel + worker errors
    svc.cancel_run()
    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("interrupted"))

    payload = last_run_finished_payload(bg.events)
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run.result is None
    assert not state.is_tab_running(tab_id)


def test_bg_done_without_cancel_reports_finished():
    state, tab_id, adapter = make_state()
    svc, _gate, bg, _ = make_run_service(state)
    svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())
    result = object()

    assert bg.last_on_done is not None
    bg.last_on_done(result)

    payload = last_run_finished_payload(bg.events)
    assert payload.outcome == "finished"
    assert state.get_tab(tab_id).run.result is result


def test_bg_done_after_cancel_reports_cancelled_with_partial():
    state, tab_id, adapter = make_state()
    svc, _gate, bg, _ = make_run_service(state)
    svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    svc.cancel_run()  # sets the captured stop_event
    partial = object()
    assert bg.last_on_done is not None
    bg.last_on_done(partial)

    payload = last_run_finished_payload(bg.events)
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run.result is partial


def test_bg_done_after_cancel_and_retry_reset_reports_cancelled():
    state, tab_id, adapter = make_state()
    partial = object()

    def run_after_retry_reset(*_args: Any, context: RunContext) -> object:
        context.cancel_signal.clear_stop()
        return partial

    adapter.run.side_effect = run_after_retry_reset
    svc, _gate, bg, handles = make_run_service(state)
    token = svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    svc.cancel_run()
    bg.run_work()

    payload = last_run_finished_payload(bg.events)
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run.result is partial
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == "cancelled"


def test_bg_error_without_cancel_reports_failed():
    state, tab_id, adapter = make_state()
    svc, _gate, bg, _ = make_run_service(state)
    svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("boom"))

    payload = last_run_finished_payload(bg.events)
    assert payload.outcome == "failed"
    assert payload.error_message == "boom"


def test_bg_error_after_cancel_reports_cancelled_without_result():
    state, tab_id, adapter = make_state()
    svc, _gate, bg, _ = make_run_service(state)
    svc.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())

    svc.cancel_run()
    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("interrupted"))

    payload = last_run_finished_payload(bg.events)
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run.result is None
