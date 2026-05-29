"""Unit tests for zcu_tools.gui.registry (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    RunRequest,
    SaveDataRequest,
    WritebackRequest,
)
from zcu_tools.gui.registry import Registry


class _DummyExp:
    """Structural ExperimentProtocol stub for the Registry-level test."""

    def run(self, soc, soccfg, cfg, **kwargs):
        del soc, soccfg, cfg, kwargs
        return object()

    def save(self, filepath, result, **kwargs):
        del filepath, result, kwargs


class _DummyCfg(ExpCfgModel):
    pass


@dataclass
class _DummyAnalyzeResult:
    figure: None = None


@dataclass
class _DummyAnalyzeParams:
    threshold: float = 0.0


class _DummyAdapter(
    AbsExpAdapter[_DummyCfg, object, _DummyAnalyzeResult, _DummyAnalyzeParams]
):
    exp_cls = _DummyExp

    def cfg_spec(self) -> CfgSectionSpec:
        return CfgSectionSpec()

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:  # noqa: ARG002
        return CfgSectionValue()

    def build_exp_cfg(self, raw_cfg, req):  # noqa: ARG002
        return _DummyCfg()

    def get_analyze_params(self, result, ctx) -> _DummyAnalyzeParams:  # noqa: ARG002
        return _DummyAnalyzeParams()

    def analyze(self, req: AnalyzeRequest[object, _DummyAnalyzeParams]):  # noqa: ARG002
        return _DummyAnalyzeResult()

    def get_writeback_items(
        self,
        req: WritebackRequest[object, _DummyAnalyzeResult],  # noqa: ARG002
    ) -> Sequence[MetaDictWriteback]:
        return []

    def make_filename_stem(self, ctx) -> str:  # noqa: ARG002
        return "dummy"

    def save(self, req: SaveDataRequest[object]) -> None:  # noqa: ARG002
        pass


# ---------------------------------------------------------------------------


def test_register_and_create():
    reg = Registry()
    reg.register("dummy", _DummyAdapter)
    adapter = reg.create("dummy")
    assert isinstance(adapter, _DummyAdapter)


def test_create_unknown_raises_key_error():
    reg = Registry()
    with pytest.raises(KeyError, match="not found"):
        reg.create("no_such")


def test_register_duplicate_raises_value_error():
    reg = Registry()
    reg.register("dummy", _DummyAdapter)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("dummy", _DummyAdapter)


def test_list_names():
    reg = Registry()
    reg.register("a", _DummyAdapter)
    reg.register("b", _DummyAdapter)
    names = reg.list_names()
    assert set(names) == {"a", "b"}


def test_has_returns_true_for_registered():
    reg = Registry()
    reg.register("dummy", _DummyAdapter)
    assert reg.has("dummy")
    assert not reg.has("other")


def test_create_returns_new_instance_each_time():
    reg = Registry()
    reg.register("dummy", _DummyAdapter)
    a1 = reg.create("dummy")
    a2 = reg.create("dummy")
    assert a1 is not a2
