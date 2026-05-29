"""Stale-operation guard (Phase 92 阻擋半).

run.start / editor.commit are blocked while the change buffer still holds an
un-surfaced change affecting their dependency. The error reply drains the
buffer (tested in test_change_buffer), so an immediate retry passes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.services.remote import ControlOptions, RemoteControlService
from zcu_tools.gui.services.remote.change_categories import (
    CAT_CFG_EDITED,
    CAT_CONTEXT_CHANGED,
    CAT_DEVICE_CHANGED,
    CAT_PREDICTOR_CHANGED,
    CAT_RUN_CHANGED,
    CAT_SOC_CHANGED,
    CAT_TAB_CHANGED,
)
from zcu_tools.gui.services.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.services.remote.service import _ClientState


@pytest.fixture(autouse=True)
def _qt(qapp):  # noqa: ARG001
    yield


def _service(editor_for_tab=None):
    ctrl = MagicMock()
    ctrl.get_bus.return_value = None
    ctrl.editor_id_for_owner.side_effect = lambda tab: (editor_for_tab or {}).get(tab)
    return RemoteControlService(controller=ctrl, opts=ControlOptions(port=0))


def _state(buffer=None):
    s = _ClientState(peer="x", token_required=False)
    if buffer:
        s.change_buffer.update(buffer)
    return s


# ---------------------------------------------------------------------------
# run.start
# ---------------------------------------------------------------------------


def test_run_blocked_when_its_tab_cfg_edited():
    svc = _service(editor_for_tab={"tab-1": "editor-1"})
    state = _state({(CAT_CFG_EDITED, "editor-1"): 1})
    with pytest.raises(RemoteError) as ei:
        svc._guard_stale(state, "run.start", {"tab_id": "tab-1"})
    assert ei.value.code == ErrorCode.PRECONDITION_FAILED
    assert ei.value.reason == "stale_tab"


def test_run_blocked_when_its_tab_changed():
    svc = _service()
    state = _state({(CAT_TAB_CHANGED, "tab-1"): 1})
    with pytest.raises(RemoteError):
        svc._guard_stale(state, "run.start", {"tab_id": "tab-1"})


def test_run_not_blocked_when_other_tab_edited():
    svc = _service(editor_for_tab={"tab-1": "editor-1", "tab-2": "editor-2"})
    # Someone edited tab-2; running tab-1 must NOT be blocked (no false positive).
    state = _state({(CAT_CFG_EDITED, "editor-2"): 1})
    svc._guard_stale(state, "run.start", {"tab_id": "tab-1"})  # no raise


def test_run_not_blocked_by_derived_categories():
    # run_changed / predictor_changed are derived state, not run inputs — they
    # must NOT block (run_changed especially would self-block on our own run).
    svc = _service(editor_for_tab={"tab-1": "editor-1"})
    state = _state({(CAT_RUN_CHANGED, "tab-1"): 1, (CAT_PREDICTOR_CHANGED, None): 1})
    svc._guard_stale(state, "run.start", {"tab_id": "tab-1"})  # no raise


def test_run_passes_with_empty_buffer():
    svc = _service(editor_for_tab={"tab-1": "editor-1"})
    svc._guard_stale(_state(), "run.start", {"tab_id": "tab-1"})  # no raise


# --- run global dependencies: soc / context / device all block any run ---


def test_run_blocked_by_soc_change():
    svc = _service(editor_for_tab={"tab-1": "editor-1"})
    state = _state({(CAT_SOC_CHANGED, None): 1})
    with pytest.raises(RemoteError) as ei:
        svc._guard_stale(state, "run.start", {"tab_id": "tab-1"})
    assert ei.value.reason == "stale_tab"


def test_run_blocked_by_context_change():
    svc = _service(editor_for_tab={"tab-1": "editor-1"})
    state = _state({(CAT_CONTEXT_CHANGED, None): 1})
    with pytest.raises(RemoteError):
        svc._guard_stale(state, "run.start", {"tab_id": "tab-1"})


def test_run_blocked_by_device_change_named():
    svc = _service(editor_for_tab={"tab-1": "editor-1"})
    state = _state({(CAT_DEVICE_CHANGED, "flux_bias"): 1})
    with pytest.raises(RemoteError):
        svc._guard_stale(state, "run.start", {"tab_id": "tab-1"})


def test_run_blocked_by_device_change_global():
    svc = _service(editor_for_tab={"tab-1": "editor-1"})
    state = _state({(CAT_DEVICE_CHANGED, None): 1})
    with pytest.raises(RemoteError):
        svc._guard_stale(state, "run.start", {"tab_id": "tab-1"})


# ---------------------------------------------------------------------------
# editor.commit
# ---------------------------------------------------------------------------


def test_commit_blocked_when_its_editor_edited():
    svc = _service()
    state = _state({(CAT_CFG_EDITED, "editor-9"): 1})
    with pytest.raises(RemoteError) as ei:
        svc._guard_stale(state, "editor.commit", {"editor_id": "editor-9"})
    assert ei.value.reason == "stale_editor"


def test_commit_not_blocked_by_other_editor():
    svc = _service()
    state = _state({(CAT_CFG_EDITED, "editor-other"): 1})
    svc._guard_stale(state, "editor.commit", {"editor_id": "editor-9"})  # no raise


def test_commit_not_blocked_by_soc_or_device():
    # SoC / device do not affect lowering an ml entry, so must NOT block commit.
    svc = _service()
    state = _state({(CAT_SOC_CHANGED, None): 1, (CAT_DEVICE_CHANGED, None): 1})
    svc._guard_stale(state, "editor.commit", {"editor_id": "editor-9"})  # no raise


def test_commit_blocked_by_context_change():
    # commit lowers EvalValue against current md -> a context/md change shifts
    # the committed concrete values, so it must block (same logic as run).
    svc = _service()
    state = _state({(CAT_CONTEXT_CHANGED, None): 1})
    with pytest.raises(RemoteError) as ei:
        svc._guard_stale(state, "editor.commit", {"editor_id": "editor-9"})
    assert ei.value.reason == "stale_editor"


# ---------------------------------------------------------------------------
# unguarded methods + missing params
# ---------------------------------------------------------------------------


def test_unguarded_method_never_blocks():
    svc = _service()
    state = _state({(CAT_CFG_EDITED, "editor-1"): 5})
    svc._guard_stale(state, "tab.snapshot", {"tab_id": "tab-1"})  # no raise


def test_run_missing_tab_id_is_skipped():
    svc = _service()
    svc._guard_stale(_state(), "run.start", {})  # no raise
