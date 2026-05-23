"""Unit tests for zcu_tools.gui.registry (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.adapter import (
    AbsExpAdapter,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    ParamSpec,
)
from zcu_tools.gui.registry import Registry


class _DummyCfg(ExpCfgModel):
    pass


@dataclass
class _DummyAnalyzeResult:
    figure: None = None


class _DummyAdapter(AbsExpAdapter):
    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:  # noqa: ARG002
        return CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())

    def build_exp_cfg(self, raw_cfg, ctx):  # noqa: ARG002
        return _DummyCfg()

    def get_analyze_params(self) -> dict[str, ParamSpec]:
        return {}

    def analyze(self, result, ctx, **kw):  # noqa: ARG002
        return _DummyAnalyzeResult()

    def get_writeback_items(self, analyze_result, ctx) -> Sequence[MetaDictWriteback]:  # noqa: ARG002
        return []

    def make_filename_stem(self, ctx) -> str:  # noqa: ARG002
        return "dummy"

    def save(self, data_path, result, ctx) -> None:  # noqa: ARG002
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
