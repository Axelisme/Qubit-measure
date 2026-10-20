from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import StoredRun, load_run_data
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    NoAnalysisResult,
    NoAnalyzeParams,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui.app.measure._load_support import write_native_file
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    native_save_session,
)


class _Cfg(ExpCfgModel):
    pass


@dataclass(frozen=True)
class _LoadedResult:
    path: Path


_LoadedRecord: TypeAlias = RunRecord[_Cfg, _LoadedResult]


class _LoadExp:
    pass


class _LoadAdapter(BaseAdapter[_Cfg, _LoadedRecord, NoAnalysisResult, NoAnalyzeParams]):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE
    )
    exp_cls = _LoadExp


class _NeedsArgsExp:
    def __init__(self, required: object) -> None:
        del required


class _NeedsArgsAdapter(_LoadAdapter):
    exp_cls = _NeedsArgsExp


class _NoLoadExp:
    pass


class _NoLoadAdapter(_LoadAdapter):
    exp_cls = _NoLoadExp


class _InternalTypeErrorExp:
    def __init__(self) -> None:
        raise TypeError("constructor bug")


class _InternalTypeErrorAdapter(_LoadAdapter):
    exp_cls = _InternalTypeErrorExp


@pytest.fixture
def stored(tmp_path: Path) -> StoredRun:
    path = tmp_path / "data.h5"
    write_native_file(path)
    return load_run_data(path)


def test_base_adapter_load_reads_native_record_and_full_history(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        session.save()
        path = session.native_data_path

        restored = AnalysisSavingAdapter().restore(load_run_data(path))
        record = restored.record

        assert isinstance(record, RunRecord)
        assert isinstance(record.cfg, SavingCfg)
        assert isinstance(record.result, SavingResult)
        np.testing.assert_array_equal(record.result.frequency, [1000, 1001, 1002])
        assert restored.cfg_issue is None


@pytest.mark.parametrize("adapter_cls", [_NeedsArgsAdapter, _NoLoadAdapter])
def test_base_adapter_restore_raises_explicit_unsupported(
    adapter_cls: type[_LoadAdapter], stored: StoredRun
) -> None:
    with pytest.raises(NotImplementedError, match="does not support loading"):
        adapter_cls().restore(stored)


def test_base_adapter_restore_preserves_constructor_internal_type_error(
    stored: StoredRun,
) -> None:
    with pytest.raises(TypeError, match="constructor bug"):
        _InternalTypeErrorAdapter().restore(stored)
