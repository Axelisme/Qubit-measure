"""Minimal adapters exercising controller Run and interactive operation lifetimes."""

from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import MagicMock

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    NoAnalysisResult,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.interactive.plugin import PluginDefinition
from zcu_tools.plotting.plots import Plots


@dataclass
class AnalyzeParams:
    threshold: float = 0.0


class RecordingAdapter(
    BaseAdapter[ExpCfgModel, object, NoAnalysisResult, AnalyzeParams]
):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False, load_data=False
    )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return "sample"

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return MeasureCfgBuilder(role_factories={}).reps(100).build()

    def run(
        self,
        req: RunRequest,
        cfg: ExpCfgModel,
        *,
        context: RunContext,
    ) -> object:
        return object()

    def get_analyze_params(self, result: object, ctx: SessionEnv) -> AnalyzeParams:
        return AnalyzeParams()

    def analyze(
        self, req: AnalyzeRequest[object, AnalyzeParams], *, plots: Plots
    ) -> NoAnalysisResult:
        return NoAnalysisResult()


class InteractiveAdapter(
    BaseAdapter[ExpCfgModel, object, NoAnalysisResult, NoAnalyzeParams]
):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False, load_data=False, analysis=AnalysisMode.INTERACTIVE
    )

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return "sample"

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return MeasureCfgBuilder(role_factories={}).build()

    def make_interactive_plugin(self, req, *, plots):
        return PluginDefinition(
            "sample", 0, (), lambda state: None, lambda state: NoAnalysisResult()
        )

    def make_interactive_frontend(
        self, plugin, session, env, request_finish, request_cancel, *, plots
    ):
        return MagicMock()
