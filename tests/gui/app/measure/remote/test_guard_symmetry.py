"""Wire projection of shared admission decisions through real app dispatch.

Admission order and capture belong to services/test_guard.py. These cases cover
remote error envelopes and their precedence over wire-specific source checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.adapter import ContextReadiness
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.app.measure.services import TabAction
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
        writeback_control=ctrl.writeback_control,
        tab_control=ctrl.tab_control,
    )
    try:
        return spec.handler(adapter, handler_params)
    except ExpectedError as exc:
        raise remote_error_from_expected(exc) from exc


def test_run_start_unselected_context_symmetry(tmp_path, qapp):
    """Unselected context projects the shared Run denial to the wire."""
    ctrl = _make_controller(ContextReadiness.EMPTY, tmp_path=tmp_path)
    tab_id = ctrl.tab_control.new_tab("fake")

    decision = ctrl.admit(tab_id, TabAction.RUN)
    with pytest.raises(RemoteError) as excinfo:
        _dispatch(
            ctrl,
            "tab.run_start",
            {
                "tab_id": tab_id,
            },
        )
    assert excinfo.value.code is ErrorCode.PRECONDITION_FAILED
    assert excinfo.value.message == decision.message
    assert excinfo.value.reason == decision.reason


def test_analyze_without_run_result_symmetry(tmp_path, qapp):
    """No result projects no_run_result before analysis option validation."""
    ctrl = _make_controller(ContextReadiness.ACTIVE, tmp_path=tmp_path)
    tab_id = ctrl.tab_control.new_tab("fake")

    decision = ctrl.admit(tab_id, TabAction.ANALYZE)
    with pytest.raises(RemoteError) as excinfo:
        _dispatch(ctrl, "tab.analyze", {"tab_id": tab_id, "updates": {}})
    assert excinfo.value.code is ErrorCode.PRECONDITION_FAILED
    assert excinfo.value.reason == "no_run_result"
    assert excinfo.value.message == decision.message


_ADMISSION_REQUESTS = [
    ("tab.run_start", {}),
    ("tab.load_data", {"data_path": "missing.h5"}),
    ("tab.analyze", {"updates": {}, "run_operation_id": 999}),
    ("tab.post_analyze", {"updates": {}, "operation_id": 999}),
    ("tab.writeback_write", {"subtab_id": "analysis", "write": [{"id": "param-1"}]}),
]


@pytest.mark.parametrize(("method", "params"), _ADMISSION_REQUESTS)
@pytest.mark.parametrize("known_tab", [False, True], ids=["unknown", "known"])
def test_experiment_unavailable_is_first_wire_reason(
    tmp_path, qapp, method: str, params: dict[str, object], known_tab: bool
) -> None:
    """Catalog access wins over tab lookup, result readiness and stale sources."""
    ctrl = _make_controller(ContextReadiness.EMPTY, tmp_path=tmp_path)
    tab_id = ctrl.tab_control.new_tab("fake") if known_tab else "missing"
    ctrl.begin_shutdown(lambda: None)
    try:
        with pytest.raises(RemoteError) as excinfo:
            _dispatch(ctrl, method, {"tab_id": tab_id, **params})
        error = excinfo.value
        assert (error.code, error.message, error.reason, error.data) == (
            ErrorCode.PRECONDITION_FAILED,
            "Experiments are unavailable during reload or shutdown",
            "experiment_unavailable",
            None,
        )
    finally:
        ctrl.abort_shutdown()


@pytest.mark.parametrize(("method", "params"), _ADMISSION_REQUESTS)
def test_unknown_tab_retains_legacy_wire_tuple(
    tmp_path, qapp, method: str, params: dict[str, object]
) -> None:
    """Remote tab lookup remains invalid_params rather than a domain error."""
    ctrl = _make_controller(ContextReadiness.EMPTY, tmp_path=tmp_path)
    with pytest.raises(RemoteError) as excinfo:
        _dispatch(ctrl, method, {"tab_id": "missing", **params})
    error = excinfo.value
    assert (error.code, error.message, error.reason, error.data) == (
        ErrorCode.INVALID_PARAMS,
        "unknown tab_id: 'missing'",
        "",
        None,
    )


def test_save_no_run_result_carries_reason(tmp_path, qapp):
    """ACTIVE context, no run result: save fails with reason='no_run_result'."""
    ctrl = _make_controller(ContextReadiness.ACTIVE, tmp_path=tmp_path)
    tab_id = ctrl.tab_control.new_tab("fake")

    with pytest.raises(RemoteError) as excinfo:
        _dispatch(ctrl, "tab.save_data", {"tab_id": tab_id})
    assert excinfo.value.code is ErrorCode.PRECONDITION_FAILED
    assert excinfo.value.reason == "no_run_result"
