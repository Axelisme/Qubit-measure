from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.remote.handlers.run_save import h_tab_load_data
from zcu_tools.gui.app.measure.services.load import LoadDataError, LoadTabResultOutcome
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from ._helpers import dispatch_handler


@dataclass
class _AnalyzeParams:
    threshold: float


@pytest.mark.parametrize("analysis_error", [None, "analysis defaults unavailable"])
def test_tab_load_data_dispatch_returns_serializable_outcome(analysis_error) -> None:
    ctrl = MagicMock()
    ctrl.has_tab.return_value = True
    ctrl.load_tab_result.return_value = LoadTabResultOutcome(
        tab_id="tab-1",
        data_path="/tmp/result.hdf5",
        result_type="Result",
        has_cfg_snapshot=True,
        has_analyze_params=analysis_error is None,
        analysis_error=analysis_error,
    )
    ctrl.get_tab_snapshot.return_value = SimpleNamespace(
        interaction=SimpleNamespace(has_run_result=True),
        analysis=SimpleNamespace(
            params=_AnalyzeParams(threshold=0.25) if analysis_error is None else None
        ),
    )
    adapter = SimpleNamespace(run_analyze_control=ctrl)

    reply = h_tab_load_data(
        cast(Any, adapter), {"tab_id": "tab-1", "data_path": "/tmp/result.hdf5"}
    )

    ctrl.load_tab_result.assert_called_once_with("tab-1", "/tmp/result.hdf5")
    assert reply == {
        "tab_id": "tab-1",
        "data_path": "/tmp/result.hdf5",
        "result_type": "Result",
        "has_cfg_snapshot": True,
        "has_analyze_params": analysis_error is None,
        "source_kind": "loaded",
        "cfg_backfill": "not_applied",
        "analysis_error": analysis_error,
        "has_run_result": True,
        "analyze_params": {"threshold": 0.25} if analysis_error is None else None,
    }


def test_tab_load_data_dispatch_maps_load_data_error() -> None:
    ctrl = MagicMock()
    ctrl.has_tab.return_value = True
    ctrl.load_tab_result.side_effect = LoadDataError(
        "Cannot load this data file into the current tab.\n\nDetails: bad axes",
        reason_code="invalid_data_file",
    )
    with pytest.raises(RemoteError) as exc_info:
        dispatch_handler(
            ctrl,
            "tab.load_data",
            {"tab_id": "tab-1", "data_path": "/tmp/bad.hdf5"},
        )

    assert exc_info.value.code is ErrorCode.PRECONDITION_FAILED
    assert exc_info.value.reason == "invalid_data_file"
    assert "bad axes" in exc_info.value.message
