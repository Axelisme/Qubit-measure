"""Guard checks and Run input capture across readiness and result states."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ConfigBase, Expr
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ContextReadiness,
)
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.services import guard as guard_api
from zcu_tools.gui.app.measure.services.form_contracts import (
    CfgPreconditionError,
    CfgPreconditionReason,
)
from zcu_tools.gui.app.measure.services.guard import (
    GuardError,
    GuardService,
    RunInputs,
)
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.state import DeviceState, DeviceStatus
from zcu_tools.program.v2.modules import PulseCfg

from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.app.measure._form_fakes import make_form


class GainCfg(ConfigBase):
    gain: float


class ModuleAdmissionCfg(ConfigBase):
    asset: PulseCfg


def _make_guard(state: State) -> GuardService:
    return GuardService(state, access=ExperimentAccess())


_DEFAULT_RECORD: RunRecord[ConfigBase, object] = RunRecord(None, object())


def _make_state(
    *,
    readiness: ContextReadiness,
    soc_attached: bool = True,
    lowering_raises: bool = False,
    run_result: RunRecord[ConfigBase, object] | None = _DEFAULT_RECORD,
    analyze_result: object = object(),
    adapter: MagicMock | None = None,
) -> tuple[State, str]:
    soc = MagicMock() if soc_attached else None
    soccfg = MagicMock() if soc_attached else None
    state = State(
        SessionEnv(
            parameters=parameter_capture()
            if readiness is ContextReadiness.ACTIVE
            else None,
            soc=soc,
            soccfg=soccfg,
            readiness=readiness,
        )
    )
    tab_id = "tab-1"
    if adapter is None:
        adapter = MagicMock()
    adapter.experiment_tag = None
    adapter.capabilities = AdapterCapabilities(
        load_data=True, analysis=AnalysisMode.FIT, post_analysis=True
    )

    form = Form(GainCfg, {}) if lowering_raises else Form(ConfigBase, {})

    tab = Session(
        adapter_name="any",
        adapter=adapter,
        cfg=make_form(form, state=state),
    )
    tab.run.result = run_result
    tab.analysis.result = analyze_result
    state.add_tab(tab_id, tab)
    return state, tab_id


def test_run_admission_orders_readiness_cfg_soc_and_busy() -> None:
    state, tab_id = _make_state(
        readiness=ContextReadiness.EMPTY, soc_attached=False, lowering_raises=True
    )
    state.set_tab_analyzing(tab_id, analyzing=True)
    guard = _make_guard(state)
    decision = guard.admit(tab_id, guard_api.TabAction.RUN)
    assert not decision.allowed and decision.reason == "no_active_context"
    state.set_context(replace(state.session_env, readiness=ContextReadiness.ACTIVE))
    decision = guard.admit(tab_id, guard_api.TabAction.RUN)
    assert not decision.allowed and decision.reason == "not_valid"
    assert decision.message.startswith("gain: ")
    state.set_tab_analyzing(tab_id, analyzing=False)
    state.get_tab(tab_id).cfg.write({("gain",): 0.25})
    state.set_tab_analyzing(tab_id, analyzing=True)
    assert guard.admit(tab_id, guard_api.TabAction.RUN).reason == "no_soc"
    state.set_context(replace(state.session_env, soc=MagicMock(), soccfg=MagicMock()))
    assert guard.admit(tab_id, guard_api.TabAction.RUN).reason == "tab_busy"
    state.set_tab_analyzing(tab_id, analyzing=False)
    # No storage requirement, no hardware exclusion for startup or another Run.
    state.set_startup_active(active=True)
    state.add_tab(
        "other",
        Session(
            adapter_name="other",
            adapter=MagicMock(),
            cfg=make_form(Form(ConfigBase, {}), state=state),
        ),
    )
    state.set_tab_running("other", running=True)
    assert guard.admit(tab_id, guard_api.TabAction.RUN).allowed


@pytest.mark.parametrize("action", list(guard_api.TabAction))
@pytest.mark.parametrize("blocked_by", ["switching", "available", "shutting_down"])
def test_experiment_access_precedes_unknown_tab_for_every_action(
    action: guard_api.TabAction, blocked_by: str
) -> None:
    state, _ = _make_state(readiness=ContextReadiness.EMPTY)
    access = ExperimentAccess()
    setattr(access, blocked_by, blocked_by != "available")
    guard = GuardService(state, access=access)
    decision = guard.admit("unknown", action)
    assert not decision.allowed and decision.reason == "experiment_unavailable"
    assert decision.message == "Experiments are unavailable during reload or shutdown"
    setattr(access, blocked_by, blocked_by == "available")
    assert guard.admit("unknown", action).reason == "unknown_tab"


def test_writeback_admission_orders_context_primary_and_busy() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.EMPTY, analyze_result=None)
    state.set_tab_analyzing(tab_id, analyzing=True)
    guard = _make_guard(state)
    assert guard.admit(tab_id, guard_api.TabAction.WRITEBACK).reason == "no_context"
    state.set_context(replace(state.session_env, parameters=parameter_capture()))
    decision = guard.admit(tab_id, guard_api.TabAction.WRITEBACK)
    assert not decision.allowed and decision.reason == "no_analyze_result"
    assert decision.message == "No analyze result available for writeback."
    with pytest.raises(GuardError) as caught:
        guard.check_writeback(tab_id)
    assert caught.value.reason_code == decision.reason
    assert str(caught.value) == decision.message
    state.get_tab(tab_id).analysis.result = object()
    assert guard.admit(tab_id, guard_api.TabAction.WRITEBACK).reason == "tab_busy"
    state.set_tab_analyzing(tab_id, analyzing=False)
    assert guard.admit(tab_id, guard_api.TabAction.WRITEBACK).allowed


def test_post_admission_orders_primary_capability_and_busy() -> None:
    adapter = MagicMock()
    state, tab_id = _make_state(
        readiness=ContextReadiness.EMPTY, analyze_result=None, adapter=adapter
    )
    tab = state.get_tab(tab_id)
    adapter.capabilities = AdapterCapabilities(post_analysis=False)
    state.set_tab_analyzing(tab_id, analyzing=True)
    guard = _make_guard(state)
    decision = guard.admit(tab_id, guard_api.TabAction.POST_ANALYZE)
    assert not decision.allowed and decision.reason == "no_analyze_result"
    assert (
        decision.message
        == f"Tab {tab_id!r} has no primary analyze result to post-analyze"
    )
    tab.analysis.result = object()
    assert (
        guard.admit(tab_id, guard_api.TabAction.POST_ANALYZE).reason
        == "no_post_analysis"
    )
    adapter.capabilities = AdapterCapabilities(post_analysis=True)
    assert guard.admit(tab_id, guard_api.TabAction.POST_ANALYZE).reason == "tab_busy"
    state.set_tab_analyzing(tab_id, analyzing=False)
    assert guard.admit(tab_id, guard_api.TabAction.POST_ANALYZE).allowed


def test_analyze_admission_orders_result_capability_and_busy() -> None:
    adapter = MagicMock()
    state, tab_id = _make_state(
        readiness=ContextReadiness.EMPTY, run_result=None, adapter=adapter
    )
    tab = state.get_tab(tab_id)
    adapter.capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)
    state.set_tab_analyzing(tab_id, analyzing=True)
    guard = _make_guard(state)
    assert guard.admit(tab_id, guard_api.TabAction.ANALYZE).reason == "no_run_result"
    tab.run.result = _DEFAULT_RECORD
    decision = guard.admit(tab_id, guard_api.TabAction.ANALYZE)
    assert not decision.allowed and decision.reason == "no_analysis"
    with pytest.raises(GuardError) as caught:
        guard.check_analyze(tab_id)
    assert caught.value.reason_code == decision.reason
    assert str(caught.value) == decision.message
    adapter.capabilities = AdapterCapabilities(analysis=AnalysisMode.FIT)
    assert guard.admit(tab_id, guard_api.TabAction.ANALYZE).reason == "tab_busy"
    state.set_tab_analyzing(tab_id, analyzing=False)
    assert guard.admit(tab_id, guard_api.TabAction.ANALYZE).allowed


def test_load_admission_requires_known_idle_tab_not_storage_or_context() -> None:
    state, tab_id = _make_state(
        readiness=ContextReadiness.EMPTY,
        soc_attached=False,
        run_result=None,
        analyze_result=None,
    )
    guard = _make_guard(state)
    decision = guard.admit(tab_id, guard_api.TabAction.LOAD)
    assert decision.allowed and decision.reason is None and decision.message == ""
    state.set_tab_analyzing(tab_id, analyzing=True)
    decision = guard.admit(tab_id, guard_api.TabAction.LOAD)
    assert not decision.allowed
    assert decision.reason == "tab_busy"
    assert decision.message == f"Tab {tab_id!r} is busy"
    unknown = guard.admit("unknown", guard_api.TabAction.LOAD)
    assert not unknown.allowed and unknown.reason == "unknown_tab"
    with pytest.raises(GuardError) as caught:
        guard.check_load(tab_id)
    assert caught.value.reason_code == decision.reason
    assert str(caught.value) == decision.message


# ---------------------------------------------------------------------------
# Run inputs
# ---------------------------------------------------------------------------


def test_capture_keeps_execution_time_notification_error_after_readonly_admission() -> (
    None
):
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    cfg = make_form(Form(GainCfg, {"gain": 0.25}), state=state)
    state.get_tab(tab_id).cfg = cfg
    guard = _make_guard(state)
    observed: list[CfgPreconditionReason] = []

    def notified() -> None:
        assert guard.admit(tab_id, guard_api.TabAction.RUN).allowed
        with pytest.raises(CfgPreconditionError) as caught:
            guard.capture_run_inputs(tab_id)
        observed.append(caught.value.reason)

    cfg.subscribe(notified)
    cfg.write({("gain",): 0.5})
    assert observed == [CfgPreconditionReason.REENTRANT_MUTATION]
    assert guard.capture_run_inputs(tab_id).accepted_cfg.values == {"gain": 0.5}


def test_run_inputs_issued_for_active_valid_cfg():
    adapter = MagicMock()
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, adapter=adapter)
    guard = _make_guard(state)

    inputs = guard.capture_run_inputs(tab_id)

    assert isinstance(inputs, RunInputs)
    assert inputs.tab_id == tab_id
    assert inputs.accepted_cfg.values == {}
    assert inputs.request.soc is state.session_env.soc
    parameters = inputs.parameters
    assert parameters is not None
    assert inputs.request.cfg_context == parameters.library.validation_context()
    assert inputs.adapter is state.get_tab(tab_id).adapter


def test_run_inputs_freezes_published_expression_and_provenance() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    state.set_context(
        replace(state.session_env, parameters=parameter_capture(gain=0.25))
    )
    cfg = make_form(Form(GainCfg, {"gain": Expr("general.ext.gain")}), state=state)
    state.get_tab(tab_id).cfg = cfg
    before = cfg.accept()
    inputs = _make_guard(state).capture_run_inputs(tab_id)
    state.set_context(
        replace(state.session_env, parameters=parameter_capture(gain=0.75))
    )
    state.version.bump("context")

    assert inputs.accepted_cfg.values == {"gain": 0.25}
    assert inputs.accepted_cfg.source_basis == before.source_basis
    assert cfg.accept().values == {"gain": 0.25}


def test_run_inputs_rejects_invalid_expression() -> None:
    adapter = MagicMock()
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, adapter=adapter)
    cfg = make_form(
        Form(GainCfg, {"gain": Expr("general.ext.missing_gain")}), state=state
    )
    state.get_tab(tab_id).cfg = cfg
    decision = _make_guard(state).admit(tab_id, guard_api.TabAction.RUN)
    with pytest.raises(GuardError) as caught:
        _make_guard(state).capture_run_inputs(tab_id)
    assert caught.value.reason_code == decision.reason == "not_valid"
    assert str(caught.value) == decision.message


def test_run_inputs_detaches_reference_shape_and_values() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    cfg = make_form(Form(ModuleAdmissionCfg, {}), state=state)
    state.get_tab(tab_id).cfg = cfg
    cfg.write(
        {
            ("asset",): {
                "kind": "inline",
                "module": {
                    "type": "pulse",
                    "ch": 0,
                    "nqz": 1,
                    "gain": 0.25,
                    "freq": 4000,
                    "phase": 0,
                    "waveform": {
                        "kind": "inline",
                        "module": {"style": "const", "length": 0.4},
                    },
                },
            },
        }
    )
    inputs = _make_guard(state).capture_run_inputs(tab_id)
    cfg.write({("asset", "module", "gain"): 0.75})
    payload = inputs.accepted_cfg.values["asset"]
    assert isinstance(payload, dict)
    assert payload["gain"] == 0.25


@pytest.mark.parametrize(
    "status",
    [
        DeviceStatus.CONNECTED,
        DeviceStatus.CONNECTING,
        DeviceStatus.DISCONNECTING,
        DeviceStatus.SETTING_UP,
    ],
)
@pytest.mark.parametrize("cached", ["missing", "invalid"])
def test_run_inputs_admits_live_devices_without_valid_cached_settings(
    status: DeviceStatus,
    cached: Literal["missing", "invalid"],
) -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    info = (
        None
        if cached == "missing"
        else BaseDeviceInfo(address="fake", type="unsupported")
    )
    state.put_device(
        DeviceState(
            name="bias",
            type_name="FakeDevice",
            address="fake",
            status=status,
            remember=False,
            info=info,
        )
    )
    inputs: RunInputs | None = None
    admission_error: GuardError | None = None
    try:
        inputs = _make_guard(state).capture_run_inputs(
            tab_id,
        )
    except GuardError as error:
        admission_error = error
    assert admission_error is None
    assert inputs is not None
    assert inputs.accepted_cfg.values == {}
    assert inputs.request.soc is state.session_env.soc


@pytest.mark.parametrize("readiness", [ContextReadiness.EMPTY])
def test_run_inputs_rejected_when_not_active(readiness: ContextReadiness):
    state, tab_id = _make_state(readiness=readiness)
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="active file-backed context") as exc:
        guard.capture_run_inputs(tab_id)
    assert exc.value.reason_code == "no_active_context"


def test_run_inputs_rejected_on_invalid_cfg():
    adapter = MagicMock()
    state, tab_id = _make_state(
        readiness=ContextReadiness.ACTIVE, lowering_raises=True, adapter=adapter
    )
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="gain: ") as caught:
        guard.capture_run_inputs(tab_id)
    assert caught.value.reason_code == "not_valid"


def test_run_inputs_rejected_when_soc_required_but_missing():
    adapter = MagicMock()
    state, tab_id = _make_state(
        readiness=ContextReadiness.ACTIVE,
        soc_attached=False,
        adapter=adapter,
    )
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="soc"):
        guard.capture_run_inputs(tab_id)


def test_run_inputs_rejected_for_unknown_tab():
    state, _ = _make_state(readiness=ContextReadiness.ACTIVE)
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="Unknown tab"):
        guard.capture_run_inputs("does-not-exist")


# ---------------------------------------------------------------------------
# Save check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Load check (no live context or SoC/result requirement)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", list(ContextReadiness))
def test_check_load_issued_without_live_context_dependency_without_soc_or_result(
    readiness: ContextReadiness,
):
    state, tab_id = _make_state(
        readiness=readiness,
        soc_attached=False,
        run_result=None,
        analyze_result=None,
    )
    guard = _make_guard(state)

    assert guard.check_load(tab_id) is None


# ---------------------------------------------------------------------------
# Analyze check (no live context dependency)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", list(ContextReadiness))
def test_check_analyze_issued_without_live_context_dependency_and_result(
    readiness: ContextReadiness,
):
    state, tab_id = _make_state(readiness=readiness)
    guard = _make_guard(state)

    assert guard.check_analyze(tab_id) is None


def test_check_analyze_rejected_without_run_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, run_result=None)
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="No run result"):
        guard.check_analyze(tab_id)


# ---------------------------------------------------------------------------
# Writeback check (requires a selected point)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", list(ContextReadiness))
def test_check_writeback_requires_selected_point_and_analyze(
    readiness: ContextReadiness,
):
    state, tab_id = _make_state(readiness=readiness)
    guard = _make_guard(state)

    if readiness is ContextReadiness.EMPTY:
        with pytest.raises(GuardError) as caught:
            guard.check_writeback(tab_id)
        assert caught.value.reason_code == "no_context"
    else:
        assert guard.check_writeback(tab_id) is None


def test_check_writeback_rejected_without_analyze_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, analyze_result=None)
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="No analyze result"):
        guard.check_writeback(tab_id)


@pytest.mark.parametrize(
    "operation", ["check_load", "check_analyze", "check_writeback"]
)
def test_check_rejects_unknown_tab_with_original_reason(operation: str) -> None:
    state, _ = _make_state(readiness=ContextReadiness.ACTIVE)
    guard = _make_guard(state)
    with pytest.raises(GuardError) as caught:
        getattr(guard, operation)("unknown")
    assert caught.value.reason_code == "unknown_tab"
