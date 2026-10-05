from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.datafile import save_labber_data
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2_gui.measure.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.experiment.v2_gui.measure.adapters.twotone import FluxDepAdapter
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    LoadDataRequest,
    NoAnalysisResult,
    NoAnalyzeParams,
    SaveDataRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter


class _Cfg(ExpCfgModel):
    pass


@dataclass(frozen=True)
class _LoadedResult:
    path: Path


_LoadedRecord: TypeAlias = RunRecord[_Cfg, _LoadedResult]


class _LoadExp:
    def load(self, filepath: Path) -> _LoadedRecord:
        return RunRecord(cfg=None, result=_LoadedResult(path=filepath))


class _LoadAdapter(BaseAdapter[_Cfg, _LoadedRecord, NoAnalysisResult, NoAnalyzeParams]):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE
    )
    exp_cls = _LoadExp

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return MeasureCfgBuilder().build()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return "load"


class _InvalidCanonicalExp:
    def load(self, filepath: Path) -> _LoadedRecord:
        raise ValueError(f"invalid canonical data: {filepath}")


class _InvalidCanonicalAdapter(_LoadAdapter):
    exp_cls = _InvalidCanonicalExp


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

    def load(self, filepath: Path) -> _LoadedRecord:
        return RunRecord(cfg=None, result=_LoadedResult(path=filepath))


class _InternalTypeErrorAdapter(_LoadAdapter):
    exp_cls = _InternalTypeErrorExp


def _request(path: str = "/tmp/result.hdf5") -> LoadDataRequest:
    return LoadDataRequest(data_path=path, md=MagicMock(), ml=MagicMock())


def test_base_adapter_load_calls_canonical_experiment_load() -> None:
    result = _LoadAdapter().load(_request("/tmp/canonical.hdf5"))

    assert result == RunRecord(
        cfg=None, result=_LoadedResult(path=Path("/tmp/canonical.hdf5"))
    )


def test_base_adapter_load_preserves_canonical_validation_error() -> None:
    with pytest.raises(ValueError, match="invalid canonical data: /tmp/invalid.hdf5"):
        _InvalidCanonicalAdapter().load(_request("/tmp/invalid.hdf5"))


def test_flux_dep_adapter_reports_canonical_axis_error(tmp_path: Path) -> None:
    path = save_labber_data(
        str(tmp_path / "invalid_flux"),
        z=("Signal", "a.u.", np.ones((2, 2), dtype=np.complex128)),
        axes=[
            ("Frequency", "Hz", [4.3e9, 4.4e9]),
            ("Wrong flux axis", "a.u.", [0.0, 1.0]),
        ],
    )

    with pytest.raises(ValueError, match="canonical axis 1 label"):
        FluxDepAdapter().load(_request(path))


@pytest.mark.parametrize("adapter_cls", [_NeedsArgsAdapter, _NoLoadAdapter])
def test_base_adapter_load_raises_explicit_unsupported(adapter_cls: type[_LoadAdapter]):
    with pytest.raises(NotImplementedError, match="does not support loading"):
        adapter_cls().load(_request())


def test_base_adapter_load_preserves_constructor_internal_type_error() -> None:
    with pytest.raises(TypeError, match="constructor bug"):
        _InternalTypeErrorAdapter().load(_request())


@pytest.mark.parametrize("missing_cfg", [False, True])
def test_base_adapter_save_passes_explicit_record_to_override(
    tmp_path: Path, missing_cfg: bool
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
    request = SaveDataRequest(
        run_result=source,
        data_path=str(destination),
        md=MagicMock(),
        ml=MagicMock(),
        chip_name="chip",
        qub_name="qubit",
        res_name="resonator",
        active_label="data",
        comment="override metadata",
    )

    Adapter().save(request)

    assert received == [(source, destination, "override metadata")]
    assert received[0][0] is source
