from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.twotone import FluxDepAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ExpContext,
    LoadDataRequest,
    NoAnalysisResult,
    NoAnalyzeParams,
)
from zcu_tools.utils.datasaver import save_labber_data


class _Cfg(ExpCfgModel):
    pass


@dataclass(frozen=True)
class _LoadedResult:
    path: str


class _LoadExp:
    last_path: ClassVar[str | None] = None

    def load(self, filepath: str) -> _LoadedResult:
        type(self).last_path = filepath
        return _LoadedResult(path=filepath)


class _LoadAdapter(BaseAdapter[_Cfg, _LoadedResult, NoAnalysisResult, NoAnalyzeParams]):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE
    )
    exp_cls = _LoadExp

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return MeasureCfgBuilder().build()

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return "load"


class _InvalidCanonicalExp:
    def load(self, filepath: str) -> _LoadedResult:
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

    def load(self, filepath: str) -> _LoadedResult:
        return _LoadedResult(path=filepath)


class _InternalTypeErrorAdapter(_LoadAdapter):
    exp_cls = _InternalTypeErrorExp


def _request(path: str = "/tmp/result.hdf5") -> LoadDataRequest:
    return LoadDataRequest(data_path=path, md=MagicMock(), ml=MagicMock())


def test_base_adapter_load_calls_canonical_experiment_load() -> None:
    result = _LoadAdapter().load(_request("/tmp/canonical.hdf5"))

    assert result == _LoadedResult(path="/tmp/canonical.hdf5")
    assert _LoadExp.last_path == "/tmp/canonical.hdf5"


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
