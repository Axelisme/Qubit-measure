"""Structural adapter fakes shared by GUI tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import RunMetadata
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalyzeRequest,
    LoadDataRequest,
    NoAnalyzeParams,
    ParameterWriteback,
    PostWritebackRequest,
    RunRequest,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.artifacts import AnalysisItem, ArtifactKey, DataItem
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots


class DummyExp:
    """Structural ExperimentProtocol stub for the Registry-level test."""

    def run(self, cfg, *, context: RunContext):
        del cfg, context
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
    experiment_tag: ClassVar[str | None] = None
    exp_cls = DummyExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[ConfigBase]:
        del ctx
        return Form(ExpCfgModel, {})

    @classmethod
    def guide(cls) -> AdapterGuide:
        return AdapterGuide(markdown="dummy", resource_dir=None)

    @classmethod
    def analyze_params_cls(cls) -> type:
        return DummyAnalyzeParams

    def get_analyze_params(self, result, ctx) -> DummyAnalyzeParams:
        del result, ctx
        return DummyAnalyzeParams()

    def validate_run_request(self, req: RunRequest, cfg: ConfigBase) -> None:
        del req, cfg

    def run(
        self,
        req: RunRequest,
        cfg: ConfigBase,
        *,
        context: RunContext,
    ):
        del req, cfg, context
        return object()

    def load(self, req: LoadDataRequest) -> tuple[object, RunMetadata]:
        del req
        raise NotImplementedError("DummyAdapter does not load native runs")

    def analyze(self, req: AnalyzeRequest[object, DummyAnalyzeParams], *, plots: Plots):
        del req, plots
        return DummyAnalyzeResult()

    def make_interactive_plugin(
        self, req: AnalyzeRequest[object, DummyAnalyzeParams], *, plots: Plots
    ):
        raise NotImplementedError

    def make_interactive_frontend(
        self, plugin, session, env, request_finish, request_cancel, *, plots: Plots
    ):
        raise NotImplementedError

    def get_writeback_items(
        self,
        req: WritebackRequest[object, DummyAnalyzeResult],
    ) -> Sequence[ParameterWriteback]:
        del req
        return []

    # -- post-analysis stubs (mirror BaseAdapter raising defaults) -----------

    @classmethod
    def post_analyze_params_cls(cls) -> type:
        # No annotated return on get_post_analyze_params → fall back to the
        # same sentinel BaseAdapter returns when reflection finds nothing.
        return NoAnalyzeParams

    def get_post_analyze_params(self, analyze_result: object, ctx: SessionEnv) -> None:
        del analyze_result, ctx
        raise NotImplementedError

    def post_analyze(self, req: object, *, plots: Plots) -> None:
        del req, plots
        raise NotImplementedError

    def get_post_writeback_items(
        self,
        req: PostWritebackRequest[object, DummyAnalyzeResult, DummyAnalyzeResult],
    ) -> Sequence[ParameterWriteback]:
        del req
        return []

    def analysis_item(
        self,
        key: ArtifactKey,
        result: object,
        parameters: object | None,
        *,
        pane: Literal["analysis", "post_analysis"],
    ) -> AnalysisItem:
        """Reject native analysis persistence unsupported by this collaborator."""
        raise NotImplementedError("DummyAdapter has no native analysis document")

    def data_item(self, key: ArtifactKey, result: object, *, comment: str) -> DataItem:
        """Reject native run persistence unsupported by this collaborator."""
        raise NotImplementedError("DummyAdapter has no native run evidence")
