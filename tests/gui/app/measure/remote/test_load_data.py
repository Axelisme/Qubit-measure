from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter
from zcu_tools.gui.app.measure.remote.handlers.run_save import h_tab_load_data
from zcu_tools.gui.app.measure.services.load import LoadDataError, LoadTabResultOutcome
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from tests.experiment._saving_support import SavingCore, SavingResult
from tests.gui.app.measure._load_support import write_native_file
from tests.gui.app.measure._native_save_support import SavingAdapter
from tests.gui.app.measure._record_support import require_record

from ._helpers import Fixture, dispatch_handler

StoredAdapter = Adapter(
    spec=replace(SavingCore.AXES_SPEC, tag="test/load", cfg_schema_version="2.0"),
    make_form=SavingAdapter.make_form,
    run=SavingAdapter.run,
    guide=None,
)


@pytest.fixture
def stored_app(qapp, tmp_path: Path):
    fx = Fixture(project_root=str(tmp_path), empty_project=True, headless=True)
    fx.registry.register("stored", StoredAdapter)
    tab_id = fx.ctrl.new_tab("stored")
    path = tmp_path / "data.h5"
    metadata = write_native_file(path)
    try:
        yield fx, tab_id, path, metadata
    finally:
        fx.stop()


def test_registered_adapter_loads_native_record_and_cfg_diagnostic(stored_app) -> None:
    fx, tab_id, path, metadata = stored_app
    tab = fx.state.get_tab(tab_id)
    before = tab.cfg.input.data
    env = tab.cfg.env
    try:
        reply = dispatch_handler(
            fx.ctrl, "tab.load_data", {"tab_id": tab_id, "data_path": str(path)}
        )
    except RemoteError as exc:
        reply = {"error": exc.reason}

    assert "/cfg: schema major" in str(reply["cfg_issue"]), reply
    assert reply["result_type"] == "RunRecord"
    record = require_record(tab.run)
    assert record.cfg is None
    assert isinstance(record.result, SavingResult)
    np.testing.assert_array_equal(record.result.frequency, [1000.0, 1001.0, 1002.0])
    np.testing.assert_array_equal(record.result.signal, [1 + 2j, 3 - 4j, 5 + 6j])
    assert record.metadata == metadata
    assert tab.run.source_path == str(path)
    assert tab.cfg.input.data == before
    assert tab.cfg.env is env


@pytest.mark.parametrize("source", ["tag", "malformed", "missing"])
def test_registered_load_rejects_file_before_adapter_and_keeps_result(
    stored_app, source: str
) -> None:
    fx, tab_id, path, _metadata = stored_app
    if source == "tag":
        path = path.with_name("other.h5")
        write_native_file(path, tag="other/experiment")
    elif source == "malformed":
        path.write_bytes(b"not native HDF5")
    else:
        path.unlink()
    tab = fx.state.get_tab(tab_id)
    old = RunRecord(None, object())
    fx.state.update_tab_result(tab_id, old)
    previous = tab.run
    with pytest.raises(RemoteError) as caught:
        dispatch_handler(
            fx.ctrl, "tab.load_data", {"tab_id": tab_id, "data_path": str(path)}
        )

    assert caught.value.reason == (
        "invalid_data_file" if source == "tag" else "data_file_read_failed"
    )
    assert tab.run is previous


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
        cfg_issue=None,
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
        # tolerated(cast): The control-only namespace is not a RemoteControlAdapter.
        cast(Any, adapter),
        {"tab_id": "tab-1", "data_path": "/tmp/result.hdf5"},
    )

    ctrl.load_tab_result.assert_called_once_with("tab-1", "/tmp/result.hdf5")
    assert reply == {
        "tab_id": "tab-1",
        "data_path": "/tmp/result.hdf5",
        "result_type": "Result",
        "cfg_issue": None,
        "has_analyze_params": analysis_error is None,
        "source_kind": "loaded",
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
