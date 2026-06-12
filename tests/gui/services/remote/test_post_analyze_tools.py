"""Post-analysis dispatch handlers (Phase 4 dual-end RPC).

Drives the post_analyze.start / tab.get_post_analyze_params /
tab.get_post_analyze_result handlers against a mock Controller, mirroring the
analyze trio. The success path spies start_post_analyze without a full
run+analyze pipeline; the gate path asserts the same fast-fail the carrier layer
enforces (no primary analyze result -> precondition_failed).
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.adapters.singleshot.ge import GEPostAnalyzeParams
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from ._helpers import dispatch_handler as _dispatch


def _snapshot(*, has_analyze_result: bool, post_params: object) -> SimpleNamespace:
    """A minimal TabSnapshot stand-in carrying the two fields the post handlers
    read: interaction.has_analyze_result and post_analyze_params."""
    return SimpleNamespace(
        interaction=SimpleNamespace(has_analyze_result=has_analyze_result),
        post_analyze_params=post_params,
    )


_DEFAULT = object()


def _ctrl(*, has_analyze_result: bool = True, post_params: object = _DEFAULT):
    if post_params is _DEFAULT:
        post_params = GEPostAnalyzeParams(backend="pca", angle=None)
    ctrl = MagicMock()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = _snapshot(
        has_analyze_result=has_analyze_result, post_params=post_params
    )
    ctrl.start_post_analyze.return_value = 77
    return ctrl


# ---------------------------------------------------------------------------
# post_analyze.start
# ---------------------------------------------------------------------------


def test_start_with_primary_result_starts_op():
    """A primary analyze result present -> the post op is started and its
    operation_id returned; updates are applied onto the param instance."""
    ctrl = _ctrl()
    res = _dispatch(
        ctrl, "post_analyze.start", {"tab_id": "t", "updates": {"backend": "center"}}
    )
    assert res == {"operation_id": 77}
    ctrl.start_post_analyze.assert_called_once()
    args, _ = ctrl.start_post_analyze.call_args
    assert args[0] == "t"
    # dataclasses.replace applied the update onto the snapshot's param instance.
    assert args[1] == GEPostAnalyzeParams(backend="center", angle=None)


def test_start_without_updates_uses_snapshot_params():
    ctrl = _ctrl()
    res = _dispatch(ctrl, "post_analyze.start", {"tab_id": "t", "updates": {}})
    assert res == {"operation_id": 77}
    args, _ = ctrl.start_post_analyze.call_args
    assert args[1] == GEPostAnalyzeParams(backend="pca", angle=None)


def test_start_without_primary_result_fast_fails():
    """No primary analyze result -> precondition_failed with the true reason,
    before the downstream 'no post params' check (mirrors analyze.start)."""
    ctrl = _ctrl(has_analyze_result=False)
    with pytest.raises(RemoteError) as excinfo:
        _dispatch(ctrl, "post_analyze.start", {"tab_id": "t", "updates": {}})
    assert excinfo.value.code is ErrorCode.PRECONDITION_FAILED
    assert excinfo.value.reason == "no_analyze_result"
    ctrl.start_post_analyze.assert_not_called()


def test_start_unknown_tab_rejected():
    ctrl = _ctrl()
    ctrl.has_tab.return_value = False
    with pytest.raises(RemoteError) as excinfo:
        _dispatch(ctrl, "post_analyze.start", {"tab_id": "nope", "updates": {}})
    assert excinfo.value.code is ErrorCode.INVALID_PARAMS


def test_start_invalid_update_field_rejected():
    """An update key the param dataclass does not have -> invalid_params (the
    dataclasses.replace TypeError is translated, not leaked)."""
    ctrl = _ctrl()
    with pytest.raises(RemoteError) as excinfo:
        _dispatch(ctrl, "post_analyze.start", {"tab_id": "t", "updates": {"nope": 1}})
    assert excinfo.value.code is ErrorCode.INVALID_PARAMS
    ctrl.start_post_analyze.assert_not_called()


# ---------------------------------------------------------------------------
# tab.get_post_analyze_params
# ---------------------------------------------------------------------------


def test_get_params_serializes_dataclass():
    ctrl = _ctrl()
    res = _dispatch(ctrl, "tab.get_post_analyze_params", {"tab_id": "t"})
    assert res["post_analyze_params"] == asdict(
        GEPostAnalyzeParams(backend="pca", angle=None)
    )


def test_get_params_none_when_absent():
    ctrl = _ctrl(post_params=None)
    res = _dispatch(ctrl, "tab.get_post_analyze_params", {"tab_id": "t"})
    assert res == {"post_analyze_params": None}


# ---------------------------------------------------------------------------
# tab.get_post_analyze_result
# ---------------------------------------------------------------------------


def test_get_result_summarizes():
    ctrl = _ctrl()
    result = MagicMock()
    result.to_summary_dict.return_value = {"fidelity": 0.97, "backend": "pca"}
    ctrl.get_post_analyze_result.return_value = result
    res = _dispatch(ctrl, "tab.get_post_analyze_result", {"tab_id": "t"})
    assert res == {"summary": {"fidelity": 0.97, "backend": "pca"}}


def test_get_result_none_when_absent():
    ctrl = _ctrl()
    ctrl.get_post_analyze_result.return_value = None
    res = _dispatch(ctrl, "tab.get_post_analyze_result", {"tab_id": "t"})
    assert res == {"summary": None}
