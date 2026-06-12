"""Unit tests for zcu_tools.gui.app.main.registry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import pytest
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    SaveDataRequest,
    WritebackRequest,
)
from zcu_tools.gui.app.main.registry import Registry


class _DummyExp:
    """Structural ExperimentProtocol stub for the Registry-level test."""

    def run(self, soc, soccfg, cfg, **kwargs):
        del soc, soccfg, cfg, kwargs
        return object()

    def save(self, filepath, result, **kwargs):
        del filepath, result, kwargs


@dataclass
class _DummyAnalyzeResult:
    figure: None = None

    def to_summary_dict(self) -> dict[str, object]:
        return {}


@dataclass
class _DummyAnalyzeParams:
    threshold: float = 0.0


class _DummyAdapter:
    """Self-contained ExpAdapterProtocol implementer for the Registry test.

    Defines every Protocol member directly (no BaseAdapter dependency), so the
    registry test stays scoped to the gui-side structural contract.
    """

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities()
    exp_cls = _DummyExp

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec()

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior="dummy",
            expects_md="dummy",
            expects_ml="dummy",
            typical_writeback="dummy",
            recommended="dummy",
        )

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:  # noqa: ARG002
        return CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())

    @classmethod
    def analyze_params_cls(cls) -> type:
        return _DummyAnalyzeParams

    def get_analyze_params(self, result, ctx) -> _DummyAnalyzeParams:  # noqa: ARG002
        return _DummyAnalyzeParams()

    def run(self, req: RunRequest, schema: CfgSchema):  # noqa: ARG002
        return object()

    def analyze(self, req: AnalyzeRequest[object, _DummyAnalyzeParams]):  # noqa: ARG002
        return _DummyAnalyzeResult()

    def setup_interactive_analysis(
        self,
        req: AnalyzeRequest[object, _DummyAnalyzeParams],  # noqa: ARG002
        host: object,  # noqa: ARG002
    ):
        raise NotImplementedError

    def get_writeback_items(
        self,
        req: WritebackRequest[object, _DummyAnalyzeResult],  # noqa: ARG002
    ) -> Sequence[MetaDictWriteback]:
        return []

    def make_save_paths(self, ctx: ExpContext):  # noqa: ARG002
        raise NotImplementedError

    # -- post-analysis stubs (mirror BaseAdapter raising defaults) -----------

    @classmethod
    def post_analyze_params_cls(cls) -> type:
        # No annotated return on get_post_analyze_params → fall back to the
        # same sentinel BaseAdapter returns when reflection finds nothing.
        return NoAnalyzeParams

    def get_post_analyze_params(self, analyze_result: object, ctx: ExpContext) -> None:  # noqa: ARG002
        raise NotImplementedError

    def post_analyze(self, req: object) -> None:  # noqa: ARG002
        raise NotImplementedError

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
