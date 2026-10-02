"""Tests for GuardService Permit issuance across readiness and result states."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.device import DeviceManager, FakeDeviceInfo
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, ContextReadiness
from zcu_tools.gui.app.measure.services.guard import (
    AnalyzePermit,
    GuardError,
    GuardService,
    LoadPermit,
    RunPermit,
    SavePermit,
    WritebackPermit,
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
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from tests.gui.app.measure._cfg_fakes import make_cfg


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
    md = MetaDict(None)
    ml = ModuleLibrary(None)
    soc = MagicMock() if soc_attached else None
    soccfg = MagicMock() if soc_attached else None
    state = State(SessionEnv(md=md, ml=ml, soc=soc, soccfg=soccfg, readiness=readiness))
    tab_id = "tab-1"
    adapter = MagicMock()
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
        adapter_name="any", adapter=adapter, cfg=make_cfg(schema, state=state)
    )
    tab.run.result = run_result
    tab.analysis.result = analyze_result
    state.add_tab(tab_id, tab)
    return state, tab_id


# ---------------------------------------------------------------------------
# Run permit
# ---------------------------------------------------------------------------


def test_run_permit_issued_for_active_valid_cfg():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    guard = GuardService(state)

    permit = guard.acquire_run_permit(tab_id, expected_revision=CfgRevision(0))

    assert isinstance(permit, RunPermit)
    assert permit.tab_id == tab_id
    assert permit.accepted_cfg.values == {}
    assert permit.request.soc is state.session_env.soc
    assert permit.adapter is state.get_tab(tab_id).adapter
    adapter = cast(MagicMock, permit.adapter)
    adapter.validate_run_request.assert_called_once_with(permit.request, {})


def test_run_permit_freezes_published_expression_and_provenance() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    state.session_env.md.update(gain=0.25)
    cfg = make_cfg(
        CfgSchema(
            CfgSectionSpec(fields={"gain": ScalarSpec("Gain", float)}),
            CfgSectionValue(fields={"gain": EvalValue("gain")}),
        ),
        state=state,
    )
    state.get_tab(tab_id).cfg = cfg
    before = cfg.observe()
    permit = GuardService(state).acquire_run_permit(
        tab_id, expected_revision=before.ref.revision
    )
    state.session_env.md.update(gain=0.75)
    state.version.bump("context")
    cfg.refresh(before.ref.revision)

    assert permit.accepted_cfg.values == {"gain": 0.25}
    assert permit.accepted_cfg.ref == before.ref
    assert permit.accepted_cfg.source_basis == before.source_basis
    assert cfg.accept(cfg.observe().ref.revision).values == {"gain": 0.75}


def test_run_permit_rejects_invalid_expression_without_preflight() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    cfg = make_cfg(
        CfgSchema(
            CfgSectionSpec(fields={"gain": ScalarSpec("Gain", float)}),
            CfgSectionValue(fields={"gain": EvalValue("missing_gain")}),
        ),
        state=state,
    )
    state.get_tab(tab_id).cfg = cfg
    with pytest.raises(CfgPreconditionError):
        GuardService(state).acquire_run_permit(
            tab_id, expected_revision=cfg.observe().ref.revision
        )
    cast(
        MagicMock, state.get_tab(tab_id).adapter
    ).validate_run_request.assert_not_called()


def test_run_permit_detaches_reference_shape_and_values() -> None:
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
    permit = GuardService(state).acquire_run_permit(
        tab_id, expected_revision=cfg.observe().ref.revision
    )
    cfg.edit(cfg.observe().ref.revision, (CfgEdit(("asset", "gain"), 0.75),))
    value.fields["gain"] = DirectValue(9.0)
    shape.fields.clear()
    assert permit.accepted_cfg.values == {"asset": {"gain": 0.25}}


def test_run_permit_rejects_stale_revision_without_preflight() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    cfg = state.get_tab(tab_id).cfg
    old = cfg.observe().ref
    cfg.refresh(old.revision)
    with pytest.raises(CfgStaleError):
        GuardService(state).acquire_run_permit(tab_id, expected_revision=old.revision)
    cast(
        MagicMock, state.get_tab(tab_id).adapter
    ).validate_run_request.assert_not_called()


def test_run_permit_detaches_observed_device_settings(monkeypatch) -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    info = FakeDeviceInfo(address="fake", value=0.25)
    state.put_device(
        DeviceState(
            name="bias",
            type_name="FakeDevice",
            address="fake",
            status=DeviceStatus.CONNECTED,
            remember=False,
            info=info,
        )
    )
    live_read = MagicMock(side_effect=AssertionError("Unexpected hardware read"))
    monkeypatch.setattr(DeviceManager, "get_all_info", live_read)

    permit = GuardService(state).acquire_run_permit(
        tab_id, expected_revision=CfgRevision(0)
    )
    info.value = 0.5
    state.set_device_info("bias", FakeDeviceInfo(address="fake", value=0.75))

    captured = permit.request.device_snapshot["bias"]
    assert isinstance(captured, FakeDeviceInfo)
    assert captured.value == 0.25
    live_read.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [
        DeviceStatus.CONNECTED,
        DeviceStatus.CONNECTING,
        DeviceStatus.DISCONNECTING,
        DeviceStatus.SETTING_UP,
    ],
)
def test_run_permit_rejects_live_device_without_observed_settings(status) -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    state.put_device(
        DeviceState(
            name="bias",
            type_name="FakeDevice",
            address="fake",
            status=status,
            remember=False,
        )
    )

    with pytest.raises(GuardError, match="no observed settings"):
        GuardService(state).acquire_run_permit(tab_id, expected_revision=CfgRevision(0))
    adapter = state.get_tab(tab_id).adapter
    assert isinstance(adapter, MagicMock)
    adapter.validate_run_request.assert_not_called()


def test_run_permit_excludes_remembered_disconnected_device() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    state.put_device(
        DeviceState(
            name="bias",
            type_name="FakeDevice",
            address="fake",
            status=DeviceStatus.MEMORY_ONLY,
            remember=True,
        )
    )

    permit = GuardService(state).acquire_run_permit(
        tab_id, expected_revision=CfgRevision(0)
    )
    assert permit.request.device_snapshot == {}


def test_run_permit_translates_adapter_preflight_error() -> None:
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    adapter = cast(MagicMock, state.get_tab(tab_id).adapter)
    preflight_error = ValueError("bad preflight")
    adapter.validate_run_request.side_effect = preflight_error
    guard = GuardService(state)

    with pytest.raises(GuardError, match="Run config invalid: bad preflight") as exc:
        guard.acquire_run_permit(tab_id, expected_revision=CfgRevision(0))

    assert exc.value.reason_code == "invalid_cfg"
    assert exc.value.__cause__ is preflight_error


@pytest.mark.parametrize("readiness", [ContextReadiness.EMPTY, ContextReadiness.DRAFT])
def test_run_permit_rejected_when_not_active(readiness: ContextReadiness):
    state, tab_id = _make_state(readiness=readiness)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="active file-backed context") as exc:
        guard.acquire_run_permit(tab_id, expected_revision=CfgRevision(0))
    assert exc.value.reason_code == "no_active_context"


def test_save_permit_reason_code_no_run_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, run_result=None)
    guard = GuardService(state)

    with pytest.raises(GuardError) as exc:
        guard.acquire_save_permit(tab_id)
    assert exc.value.reason_code == "no_run_result"


def test_analyze_permit_reason_code_no_context():
    state, tab_id = _make_state(readiness=ContextReadiness.EMPTY)
    guard = GuardService(state)

    with pytest.raises(GuardError) as exc:
        guard.acquire_analyze_permit(tab_id)
    assert exc.value.reason_code == "no_context"


def test_run_permit_rejected_on_invalid_cfg():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, lowering_raises=True)
    adapter = cast(MagicMock, state.get_tab(tab_id).adapter)
    guard = GuardService(state)

    with pytest.raises(CfgPreconditionError, match="valid"):
        guard.acquire_run_permit(tab_id, expected_revision=CfgRevision(0))
    adapter.validate_run_request.assert_not_called()


def test_run_permit_rejected_when_soc_required_but_missing():
    state, tab_id = _make_state(
        readiness=ContextReadiness.ACTIVE, soc_attached=False, requires_soc=True
    )
    adapter = cast(MagicMock, state.get_tab(tab_id).adapter)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="soc"):
        guard.acquire_run_permit(tab_id, expected_revision=CfgRevision(0))
    adapter.validate_run_request.assert_not_called()


def test_run_permit_issued_without_soc_when_capability_does_not_require():
    state, tab_id = _make_state(
        readiness=ContextReadiness.ACTIVE, soc_attached=False, requires_soc=False
    )
    guard = GuardService(state)

    permit = guard.acquire_run_permit(tab_id, expected_revision=CfgRevision(0))
    assert isinstance(permit, RunPermit)


def test_run_permit_rejected_for_unknown_tab():
    state, _ = _make_state(readiness=ContextReadiness.ACTIVE)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="Unknown tab"):
        guard.acquire_run_permit("does-not-exist", expected_revision=CfgRevision(0))


# ---------------------------------------------------------------------------
# Save permit
# ---------------------------------------------------------------------------


def test_save_permit_issued_for_active_with_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    guard = GuardService(state)

    assert isinstance(guard.acquire_save_permit(tab_id), SavePermit)


@pytest.mark.parametrize("readiness", [ContextReadiness.EMPTY, ContextReadiness.DRAFT])
def test_save_permit_rejected_when_not_active(readiness: ContextReadiness):
    state, tab_id = _make_state(readiness=readiness)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="active file-backed context"):
        guard.acquire_save_permit(tab_id)


def test_save_permit_rejected_without_run_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, run_result=None)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="No run result"):
        guard.acquire_save_permit(tab_id)


# ---------------------------------------------------------------------------
# Load permit (allows DRAFT, no SoC/result requirement)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", [ContextReadiness.DRAFT, ContextReadiness.ACTIVE])
def test_load_permit_issued_with_context_without_soc_or_result(
    readiness: ContextReadiness,
):
    state, tab_id = _make_state(
        readiness=readiness,
        soc_attached=False,
        run_result=None,
        analyze_result=None,
    )
    guard = GuardService(state)

    assert isinstance(guard.acquire_load_permit(tab_id), LoadPermit)


def test_load_permit_rejected_when_empty():
    state, tab_id = _make_state(readiness=ContextReadiness.EMPTY)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="no experiment context") as exc:
        guard.acquire_load_permit(tab_id)
    assert exc.value.reason_code == "no_context"


# ---------------------------------------------------------------------------
# Analyze permit (allows DRAFT)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", [ContextReadiness.DRAFT, ContextReadiness.ACTIVE])
def test_analyze_permit_issued_with_context_and_result(
    readiness: ContextReadiness,
):
    state, tab_id = _make_state(readiness=readiness)
    guard = GuardService(state)

    assert isinstance(guard.acquire_analyze_permit(tab_id), AnalyzePermit)


def test_analyze_permit_rejected_when_empty():
    state, tab_id = _make_state(readiness=ContextReadiness.EMPTY)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="no experiment context"):
        guard.acquire_analyze_permit(tab_id)


def test_analyze_permit_rejected_without_run_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, run_result=None)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="No run result"):
        guard.acquire_analyze_permit(tab_id)


# ---------------------------------------------------------------------------
# Writeback permit (allows DRAFT)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", [ContextReadiness.DRAFT, ContextReadiness.ACTIVE])
def test_writeback_permit_issued_with_context_and_analyze(
    readiness: ContextReadiness,
):
    state, tab_id = _make_state(readiness=readiness)
    guard = GuardService(state)

    assert isinstance(guard.acquire_writeback_permit(tab_id), WritebackPermit)


def test_writeback_permit_rejected_when_empty():
    state, tab_id = _make_state(readiness=ContextReadiness.EMPTY)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="no experiment context"):
        guard.acquire_writeback_permit(tab_id)


def test_writeback_permit_rejected_without_analyze_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, analyze_result=None)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="No analyze result"):
        guard.acquire_writeback_permit(tab_id)
