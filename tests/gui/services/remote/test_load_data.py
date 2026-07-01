from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.load import LoadDataError, LoadTabResultOutcome
from zcu_tools.gui.app.main.services.remote.handlers.run_save import _h_tab_load_data
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError


@dataclass
class _AnalyzeParams:
    threshold: float


def test_tab_load_data_dispatch_returns_serializable_outcome() -> None:
    ctrl = MagicMock()
    ctrl.has_tab.return_value = True
    ctrl.load_tab_result.return_value = LoadTabResultOutcome(
        tab_id="tab-1",
        data_path="/tmp/result.hdf5",
        result_type="Result",
        has_cfg_snapshot=True,
        has_analyze_params=True,
    )
    ctrl.get_tab_snapshot.return_value = SimpleNamespace(
        interaction=SimpleNamespace(has_run_result=True),
        analyze_params=_AnalyzeParams(threshold=0.25),
    )
    adapter = SimpleNamespace(ctrl=ctrl)

    reply = _h_tab_load_data(
        cast(Any, adapter), {"tab_id": "tab-1", "data_path": "/tmp/result.hdf5"}
    )

    ctrl.load_tab_result.assert_called_once_with("tab-1", "/tmp/result.hdf5")
    assert reply == {
        "tab_id": "tab-1",
        "data_path": "/tmp/result.hdf5",
        "result_type": "Result",
        "has_cfg_snapshot": True,
        "has_analyze_params": True,
        "source_kind": "loaded",
        "has_run_result": True,
        "analyze_params": {"threshold": 0.25},
    }


def test_tab_load_data_dispatch_maps_load_data_error() -> None:
    ctrl = MagicMock()
    ctrl.has_tab.return_value = True
    ctrl.load_tab_result.side_effect = LoadDataError(
        "Cannot load this data file into the current tab.\n\nDetails: bad axes",
        reason_code="invalid_data_file",
    )
    adapter = SimpleNamespace(ctrl=ctrl)

    with pytest.raises(RemoteError) as exc_info:
        _h_tab_load_data(
            cast(Any, adapter), {"tab_id": "tab-1", "data_path": "/tmp/bad.hdf5"}
        )

    assert exc_info.value.code is ErrorCode.PRECONDITION_FAILED
    assert exc_info.value.reason == "invalid_data_file"
    assert "bad axes" in exc_info.value.message
