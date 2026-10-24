"""Guard symmetry: the remote dispatch path enforces the same domain guard as
the UI (Controller) path.

Both clients route protected operations through GuardService. This test invokes
the dispatch handlers directly against a real Controller and asserts that the
same precondition that the UI path rejects (Controller raises GuardError) is
surfaced as a RemoteError(PRECONDITION_FAILED) on the remote path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.adapter import ContextReadiness
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.app.measure.services.guard import GuardError
from zcu_tools.gui.expected_error import ExpectedError
from zcu_tools.gui.remote.errors import (
    ErrorCode,
    RemoteError,
    remote_error_from_expected,
)
from zcu_tools.gui.remote.param_spec import validate_params
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)


def _make_controller(readiness: ContextReadiness, *, tmp_path) -> Controller:
    from ._helpers import Fixture

    fixture = Fixture(empty_project=True, project_root=str(tmp_path / "project"))
    fixture.ctrl.setup_control.new_entry(EntryRequest("fake_qubit"))
    if readiness is ContextReadiness.ACTIVE:
        fixture.ctrl.setup_control.new_point(
            PointCreateRequest("ctx001", "", CommandOrigin("gui"))
        )
    fixture.state.set_context(
        replace(fixture.state.session_env, soc=MagicMock(), soccfg=MagicMock())
    )
    return fixture.ctrl


def _dispatch(ctrl: Controller, method: str, params: Mapping[str, object]) -> object:
    """Mirror the service path: validate params against the method's ParamSpec
    before invoking the handler. Handlers receive the adapter (ADR-0068), so
    wrap ctrl in a minimal adapter stub."""
    from types import SimpleNamespace

    spec = METHOD_REGISTRY[method]
    handler_params = validate_params(spec.params, params) if spec.params else params
    adapter = SimpleNamespace(
        ctrl=ctrl,
        run_analyze_control=ctrl.run_analyze_control,
        render_view=None,
        operation_control=ctrl.operation_control,
        save_control=ctrl.save_control,
        tab_control=ctrl.tab_control,
    )
    try:
        return spec.handler(adapter, handler_params)
    except ExpectedError as exc:
        raise remote_error_from_expected(exc) from exc


def test_run_start_unselected_context_symmetry(tmp_path, qapp):
    """Unselected context: UI raises GuardError; remote returns PRECONDITION_FAILED."""
    ctrl = _make_controller(ContextReadiness.EMPTY, tmp_path=tmp_path)
    tab_id = ctrl.tab_control.new_tab("fake")

    # UI path
    with pytest.raises(GuardError, match="active file-backed context"):
        ctrl.run_analyze_control.start_run(tab_id)

    # Remote path — same precondition, mapped to a typed wire error.
    with pytest.raises(RemoteError) as excinfo:
        _dispatch(
            ctrl,
            "tab.run_start",
            {
                "tab_id": tab_id,
            },
        )
    assert excinfo.value.code is ErrorCode.PRECONDITION_FAILED
    assert "active file-backed context" in excinfo.value.message


def test_analyze_without_run_result_symmetry(tmp_path, qapp):
    """ACTIVE context but no run result: both paths reject on the same guard."""
    ctrl = _make_controller(ContextReadiness.ACTIVE, tmp_path=tmp_path)
    tab_id = ctrl.tab_control.new_tab("fake")

    with pytest.raises(GuardError, match="No run result"):
        ctrl.run_analyze_control.analyze(tab_id, object())

    # The wire path (tab.analyze) checks run-result presence first too, so the
    # agent gets the true reason 'no_run_result' rather than the downstream
    # 'no analyze params available' (params only exist once a run produced one).
    with pytest.raises(RemoteError) as excinfo:
        _dispatch(ctrl, "tab.analyze", {"tab_id": tab_id, "updates": {}})
    assert excinfo.value.code is ErrorCode.PRECONDITION_FAILED
    assert excinfo.value.reason == "no_run_result"


def test_save_no_run_result_carries_reason(tmp_path, qapp):
    """ACTIVE context, no run result: save fails with reason='no_run_result'."""
    ctrl = _make_controller(ContextReadiness.ACTIVE, tmp_path=tmp_path)
    tab_id = ctrl.tab_control.new_tab("fake")

    with pytest.raises(RemoteError) as excinfo:
        _dispatch(ctrl, "tab.save_data", {"tab_id": tab_id})
    assert excinfo.value.code is ErrorCode.PRECONDITION_FAILED
    assert excinfo.value.reason == "no_run_result"
