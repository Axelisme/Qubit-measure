"""Guard checks and Run input capture across readiness and result states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Literal, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, ContextReadiness
from zcu_tools.gui.app.measure.services.guard import (
    GuardError,
    GuardService,
    RunInputs,
)
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgPreconditionError,
    CfgRevision,
    CfgStaleError,
)
from zcu_tools.gui.session.state import DeviceState, DeviceStatus
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.entry import RoleSpec

from tests.gui.app.measure._cfg_fakes import cfg_resolution, make_cfg, parameter_capture


def _make_guard(state: State) -> GuardService:
    def capture(
        roles: Mapping[str, RoleSpec], _target: str | None
    ) -> ParameterContext | None:
        parameters = state.session_env.parameters
        return None if parameters is None else parameters.bind_aliases((), roles)

    return GuardService(
        state,
        captured_parameters=capture,
        cfg_resolution=lambda parameters: cfg_resolution(state, parameters),
    )


def _make_state(
    *,
    readiness: ContextReadiness,
    soc_attached: bool = True,
    requires_soc: bool = True,
    lowering_raises: bool = False,
    run_result: object = object(),
    analyze_result: object = object(),
    load_data: bool = True,
) -> tuple[State, str]:
    soc = MagicMock() if soc_attached else None
    soccfg = MagicMock() if soc_attached else None
    state = State(
        SessionEnv(
            parameters=parameter_capture(), soc=soc, soccfg=soccfg, readiness=readiness
        )
    )
    tab_id = "tab-1"
    adapter = MagicMock()
    adapter.parameter_roles.return_value = {}
    adapter.experiment_tag = None
    adapter.capabilities = AdapterCapabilities(
        requires_soc=requires_soc, load_data=load_data
    )

    if lowering_raises:
        schema = CfgSchema(
            spec=CfgSectionSpec(fields={"gain": ScalarSpec(label="Gain", type=float)}),
            value=CfgSectionValue(fields={"gain": DirectValue(None)}),
        )
    else:
        schema = CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())

    tab = Session(
        adapter_name="any",
        adapter=adapter,
        cfg=make_cfg(schema, state=state),
    )
    tab.run.result = run_result
    tab.analysis.result = analyze_result
    state.add_tab(tab_id, tab)
    return state, tab_id


# ---------------------------------------------------------------------------
# Run inputs
# ---------------------------------------------------------------------------


def test_run_inputs_issued_for_active_valid_cfg():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    guard = _make_guard(state)

    inputs = guard.capture_run_inputs(tab_id, expected_revision=CfgRevision(0))

    assert isinstance(inputs, RunInputs)
    assert inputs.tab_id == tab_id
    assert inputs.accepted_cfg.values == {}
    assert inputs.request.soc is state.session_env.soc
    parameters = inputs.parameters
    assert parameters is not None
    assert inputs.request.cfg_context == parameters.library.validation_context()
    assert inputs.adapter is state.get_tab(tab_id).adapter
    adapter = cast(MagicMock, inputs.adapter)
    adapter.validate_run_request.assert_called_once_with(inputs.request, {})


def test_run_inputs_freezes_published_expression_and_provenance() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    state.set_context(
        replace(state.session_env, parameters=parameter_capture(gain=0.25))
    )
    cfg = make_cfg(
        CfgSchema(
            CfgSectionSpec(fields={"gain": ScalarSpec("Gain", float)}),
            CfgSectionValue(fields={"gain": EvalValue("general.ext.gain")}),
        ),
        state=state,
    )
    state.get_tab(tab_id).cfg = cfg
    before = cfg.observe()
    inputs = _make_guard(state).capture_run_inputs(
        tab_id, expected_revision=before.ref.revision
    )
    state.set_context(
        replace(state.session_env, parameters=parameter_capture(gain=0.75))
    )
    state.version.bump("context")
    cfg.refresh(before.ref.revision)

    assert inputs.accepted_cfg.values == {"gain": 0.25}
    assert inputs.accepted_cfg.ref == before.ref
    assert inputs.accepted_cfg.source_basis == before.source_basis
    assert cfg.accept(cfg.observe().ref.revision).values == {"gain": 0.25}


def test_run_inputs_rejects_invalid_expression_without_preflight() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    cfg = make_cfg(
        CfgSchema(
            CfgSectionSpec(fields={"gain": ScalarSpec("Gain", float)}),
            CfgSectionValue(fields={"gain": EvalValue("general.ext.missing_gain")}),
        ),
        state=state,
    )
    state.get_tab(tab_id).cfg = cfg
    with pytest.raises(CfgPreconditionError):
        _make_guard(state).capture_run_inputs(
            tab_id, expected_revision=cfg.observe().ref.revision
        )
    cast(
        MagicMock, state.get_tab(tab_id).adapter
    ).validate_run_request.assert_not_called()


def test_run_inputs_detaches_reference_shape_and_values() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    shape = CfgSectionSpec(
        label="Pulse", fields={"gain": ScalarSpec(label="Gain", type=float)}
    )
    value = CfgSectionValue(fields={"gain": DirectValue(0.25)})
    cfg = make_cfg(
        CfgSchema(
            CfgSectionSpec(
                fields={"asset": ReferenceSpec(kind="module", allowed=[shape])}
            ),
            CfgSectionValue(
                fields={
                    "asset": ReferenceValue(
                        "<Custom:Pulse>", value, resolved_label="Pulse"
                    )
                }
            ),
        ),
        state=state,
    )
    state.get_tab(tab_id).cfg = cfg
    inputs = _make_guard(state).capture_run_inputs(
        tab_id, expected_revision=cfg.observe().ref.revision
    )
    cfg.edit(cfg.observe().ref.revision, (CfgEdit(("asset", "gain"), 0.75),))
    value.fields["gain"] = DirectValue(9.0)
    shape.fields.clear()
    assert inputs.accepted_cfg.values == {"asset": {"gain": 0.25}}


def test_run_inputs_rejects_stale_revision_without_preflight() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    cfg = state.get_tab(tab_id).cfg
    old = cfg.observe().ref
    cfg.refresh(old.revision)
    with pytest.raises(CfgStaleError):
        _make_guard(state).capture_run_inputs(tab_id, expected_revision=old.revision)
    cast(
        MagicMock, state.get_tab(tab_id).adapter
    ).validate_run_request.assert_not_called()


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
            expected_revision=CfgRevision(0),
        )
    except GuardError as error:
        admission_error = error
    assert admission_error is None
    assert inputs is not None
    assert inputs.accepted_cfg.values == {}
    assert inputs.request.soc is state.session_env.soc


def test_run_inputs_translates_adapter_preflight_error() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    adapter = cast(MagicMock, state.get_tab(tab_id).adapter)
    preflight_error = ValueError("bad preflight")
    adapter.validate_run_request.side_effect = preflight_error
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="Run config invalid: bad preflight") as exc:
        guard.capture_run_inputs(tab_id, expected_revision=CfgRevision(0))

    assert exc.value.reason_code == "invalid_cfg"
    assert exc.value.__cause__ is preflight_error


@pytest.mark.parametrize("readiness", [ContextReadiness.EMPTY, ContextReadiness.DRAFT])
def test_run_inputs_rejected_when_not_active(readiness: ContextReadiness):
    state, tab_id = _make_state(readiness=readiness)
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="active file-backed context") as exc:
        guard.capture_run_inputs(tab_id, expected_revision=CfgRevision(0))
    assert exc.value.reason_code == "no_active_context"


def test_check_save_reason_code_no_run_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, run_result=None)
    guard = _make_guard(state)

    with pytest.raises(GuardError) as exc:
        guard.check_save(tab_id)
    assert exc.value.reason_code == "no_run_result"


def test_run_inputs_rejected_on_invalid_cfg():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, lowering_raises=True)
    adapter = cast(MagicMock, state.get_tab(tab_id).adapter)
    guard = _make_guard(state)

    with pytest.raises(CfgPreconditionError, match="valid"):
        guard.capture_run_inputs(tab_id, expected_revision=CfgRevision(0))
    adapter.validate_run_request.assert_not_called()


def test_run_inputs_rejected_when_soc_required_but_missing():
    state, tab_id = _make_state(
        readiness=ContextReadiness.ACTIVE, soc_attached=False, requires_soc=True
    )
    adapter = cast(MagicMock, state.get_tab(tab_id).adapter)
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="soc"):
        guard.capture_run_inputs(tab_id, expected_revision=CfgRevision(0))
    adapter.validate_run_request.assert_not_called()


def test_run_inputs_issued_without_soc_when_capability_does_not_require():
    state, tab_id = _make_state(
        readiness=ContextReadiness.ACTIVE, soc_attached=False, requires_soc=False
    )
    guard = _make_guard(state)

    inputs = guard.capture_run_inputs(tab_id, expected_revision=CfgRevision(0))
    assert isinstance(inputs, RunInputs)


def test_run_inputs_rejected_for_unknown_tab():
    state, _ = _make_state(readiness=ContextReadiness.ACTIVE)
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="Unknown tab"):
        guard.capture_run_inputs("does-not-exist", expected_revision=CfgRevision(0))


# ---------------------------------------------------------------------------
# Save check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", list(ContextReadiness))
def test_check_save_issued_with_result_without_live_context_dependency(
    readiness: ContextReadiness,
):
    state, tab_id = _make_state(readiness=readiness)
    guard = _make_guard(state)

    assert guard.check_save(tab_id) is None


def test_check_save_rejected_without_run_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, run_result=None)
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="No run result"):
        guard.check_save(tab_id)


# ---------------------------------------------------------------------------
# Load check (allows DRAFT, no SoC/result requirement)
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
# Analyze check (allows DRAFT)
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
# Writeback check (allows DRAFT)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", list(ContextReadiness))
def test_check_writeback_issued_without_live_context_dependency_and_analyze(
    readiness: ContextReadiness,
):
    state, tab_id = _make_state(readiness=readiness)
    guard = _make_guard(state)

    assert guard.check_writeback(tab_id) is None


def test_check_writeback_rejected_without_analyze_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, analyze_result=None)
    guard = _make_guard(state)

    with pytest.raises(GuardError, match="No analyze result"):
        guard.check_writeback(tab_id)


@pytest.mark.parametrize(
    "operation", ["check_save", "check_load", "check_analyze", "check_writeback"]
)
def test_check_rejects_unknown_tab_with_original_reason(operation: str) -> None:
    state, _ = _make_state(readiness=ContextReadiness.ACTIVE)
    guard = _make_guard(state)
    with pytest.raises(GuardError) as caught:
        getattr(guard, operation)("unknown")
    assert caught.value.reason_code == "unknown_tab"


def test_check_load_rejects_unsupported_adapter_with_original_reason() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, load_data=False)
    with pytest.raises(GuardError) as caught:
        _make_guard(state).check_load(tab_id)
    assert caught.value.reason_code == "unsupported_load"
