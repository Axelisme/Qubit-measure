from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ExpContext,
    LoadDataRequest,
    NoAnalysisResult,
    NoAnalyzeParams,
)


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


@pytest.mark.parametrize("adapter_cls", [_NeedsArgsAdapter, _NoLoadAdapter])
def test_base_adapter_load_raises_explicit_unsupported(adapter_cls: type[_LoadAdapter]):
    with pytest.raises(NotImplementedError, match="does not support loading"):
        adapter_cls().load(_request())


def test_base_adapter_load_preserves_constructor_internal_type_error() -> None:
    with pytest.raises(TypeError, match="constructor bug"):
        _InternalTypeErrorAdapter().load(_request())
