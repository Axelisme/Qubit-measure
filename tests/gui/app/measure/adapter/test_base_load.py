from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import load_run_data
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterCfgDefinition,
    AnalysisMode,
    LoadDataRequest,
    NoAnalysisResult,
    NoAnalyzeParams,
    SaveDataRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    native_save_session,
)
from tests.gui.app.measure.adapter._cfg_fakes import StaticCfgDefinition


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

    @classmethod
    def cfg_definition(cls) -> AdapterCfgDefinition:
        return StaticCfgDefinition()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return "load"


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


def _request(path: str = "/tmp/result.hdf5") -> LoadDataRequest:
    return LoadDataRequest(data_path=path, parameters=None)


def test_base_adapter_load_reads_native_record_and_full_history(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        session.save()
        path = (
            session.entry.database_path / "runs" / session.metadata.run_id / "data.h5"
        )

        record, metadata = AnalysisSavingAdapter().load(_request(str(path)))

        assert isinstance(record, RunRecord)
        assert isinstance(record.cfg, SavingCfg)
        assert isinstance(record.result, SavingResult)
        np.testing.assert_array_equal(record.result.frequency, [1000, 1001, 1002])
        assert metadata == load_run_data(path).metadata


@pytest.mark.parametrize("adapter_cls", [_NeedsArgsAdapter, _NoLoadAdapter])
def test_base_adapter_load_raises_explicit_unsupported(adapter_cls: type[_LoadAdapter]):
    with pytest.raises(NotImplementedError, match="does not support loading"):
        adapter_cls().load(_request())


def test_base_adapter_load_preserves_constructor_internal_type_error() -> None:
    with pytest.raises(TypeError, match="constructor bug"):
        _InternalTypeErrorAdapter().load(_request())


@pytest.mark.parametrize("missing_cfg", [False, True])
def test_base_adapter_save_passes_explicit_record_to_override(
    tmp_path: Path, missing_cfg: bool, qapp: QApplication
) -> None:
    received: list[tuple[_LoadedRecord, Path, str | None]] = []

    class Saver(_LoadExp):
        def save(
            self,
            source: _LoadedRecord,
            destination: Path,
            *,
            comment: str | None = None,
        ) -> None:
            received.append((source, destination, comment))

    class Adapter(_LoadAdapter):
        exp_cls = Saver

    source = RunRecord(
        cfg=None if missing_cfg else _Cfg(),
        result=_LoadedResult(path=tmp_path / "source.hdf5"),
    )
    destination = tmp_path / "exact.hdf5"
    with native_save_session(tmp_path, qapp) as native:
        request = SaveDataRequest(
            run_result=source,
            data_path=str(destination),
            parameters=native.state.get_tab("tab").run.parameters,
            storage=native.context.read_context().capture_storage(),
            comment="override metadata",
        )

        Adapter().save(request)

    assert received == [(source, destination, "override metadata")]
    assert received[0][0] is source
