"""Structural adapter fakes shared by GUI tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalyzeRequest,
    ExpContext,
    LoadDataRequest,
    MetaDictWriteback,
    NoAnalyzeParams,
    PostWritebackRequest,
    RunRequest,
    SaveDataRequest,
    WritebackRequest,
)
from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue


class DummyExp:
    """Structural ExperimentProtocol stub for the Registry-level test."""

    def run(self, soc, soccfg, cfg, **kwargs):
        del soc, soccfg, cfg, kwargs
        return object()

    def save(self, filepath, result, **kwargs):
        del filepath, result, kwargs


@dataclass
class DummyAnalyzeResult:
    figure: None = None

    def to_summary_dict(self) -> dict[str, object]:
        return {}


@dataclass
class DummyAnalyzeParams:
    threshold: float = 0.0


class DummyAdapter:
    """Self-contained ExpAdapterProtocol implementer for the Registry test.

    Defines every Protocol member directly (no BaseAdapter dependency), so the
    registry test stays scoped to the gui-side structural contract.
    """

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities()
    exp_cls = DummyExp

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(
            behavior="dummy",
            expects_md="dummy",
            expects_ml="dummy",
            typical_writeback="dummy",
            recommended="dummy",
        )

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        del ctx
        return CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())

    @classmethod
    def analyze_params_cls(cls) -> type:
        return DummyAnalyzeParams

    def get_analyze_params(self, result, ctx) -> DummyAnalyzeParams:
        del result, ctx
        return DummyAnalyzeParams()

    def validate_run_request(self, req: RunRequest, raw_cfg: dict[str, object]) -> None:
        del req, raw_cfg

    def run(self, req: RunRequest, schema: CfgSchema):
        del req, schema
        return object()

    def load(self, req: LoadDataRequest):
        del req
        return object()

    def analyze(self, req: AnalyzeRequest[object, DummyAnalyzeParams]):
        del req
        return DummyAnalyzeResult()

    def setup_interactive_analysis(
        self,
        req: AnalyzeRequest[object, DummyAnalyzeParams],
        host: object,
    ):
        del req, host
        raise NotImplementedError

    def get_writeback_items(
        self,
        req: WritebackRequest[object, DummyAnalyzeResult],
    ) -> Sequence[MetaDictWriteback]:
        del req
        return []

    def make_save_paths(self, ctx: ExpContext):
        del ctx
        raise NotImplementedError

    # -- post-analysis stubs (mirror BaseAdapter raising defaults) -----------

    @classmethod
    def post_analyze_params_cls(cls) -> type:
        # No annotated return on get_post_analyze_params → fall back to the
        # same sentinel BaseAdapter returns when reflection finds nothing.
        return NoAnalyzeParams

    def get_post_analyze_params(self, analyze_result: object, ctx: ExpContext) -> None:
        del analyze_result, ctx
        raise NotImplementedError

    def post_analyze(self, req: object) -> None:
        del req
        raise NotImplementedError

    def get_post_writeback_items(
        self,
        req: PostWritebackRequest[object, DummyAnalyzeResult, DummyAnalyzeResult],
    ) -> Sequence[MetaDictWriteback]:
        del req
        return []

    def save(self, req: SaveDataRequest[object]) -> None:
        del req
