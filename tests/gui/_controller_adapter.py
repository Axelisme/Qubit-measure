"""Minimal adapters exercising controller Run and interactive operation lifetimes."""

from dataclasses import dataclass
from unittest.mock import MagicMock

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AxesSpec, ZSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    Interactive,
    SessionEnv,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.interactive.plugin import PluginDefinition
from zcu_tools.plotting.plots import Plots


@dataclass
class AnalyzeParams:
    threshold: float = 0.0


class RecordingCfg(ConfigBase):
    reps: int = 100


@dataclass
class RecordingResult:
    value: float = 0.0


def RecordingAdapter_make_form(ctx: FormContext) -> Form[RecordingCfg]:
    del ctx
    return Form(RecordingCfg, {})


def RecordingAdapter_run(
    cfg: RecordingCfg, *, context: RunContext
) -> RunRecord[RecordingCfg, RecordingResult]:
    return RunRecord(cfg=cfg, result=RecordingResult())


def RecordingAdapter_get_analyze_params(
    result: RunRecord[RecordingCfg, RecordingResult], ctx: SessionEnv
) -> AnalyzeParams:
    return AnalyzeParams()


def RecordingAdapter_analyze(
    req: AnalyzeRequest[RunRecord[RecordingCfg, RecordingResult], AnalyzeParams],
    *,
    plots: Plots,
) -> None:
    return None


RecordingAdapter = Adapter(
    spec=AxesSpec(
        axes=(),
        z=ZSpec("value", "data", ""),
        result_type=RecordingResult,
        cfg_type=RecordingCfg,
        tag="test/RecordingAdapter",
    ),
    make_form=RecordingAdapter_make_form,
    run=RecordingAdapter_run,
    guide=None,
    analysis=Analysis(
        fn=RecordingAdapter_analyze,
        options=RecordingAdapter_get_analyze_params,
        writeback=None,
    ),
    post_analysis=None,
)


def InteractiveAdapter_make_form(ctx: FormContext) -> Form[RecordingCfg]:
    del ctx
    return Form(RecordingCfg, {})


def InteractiveAdapter_make_interactive_plugin(req, *, plots):
    return PluginDefinition("sample", 0, (), lambda state: None, lambda state: None)


def InteractiveAdapter_make_interactive_frontend(
    plugin, session, env, request_finish, request_cancel, *, plots
):
    return MagicMock()


InteractiveAdapter = Adapter(
    spec=AxesSpec(
        axes=(),
        z=ZSpec("value", "data", ""),
        result_type=RecordingResult,
        cfg_type=RecordingCfg,
        tag="test/InteractiveAdapter",
    ),
    make_form=InteractiveAdapter_make_form,
    run=RecordingAdapter.run,
    guide=None,
    analysis=Interactive(
        plugin=InteractiveAdapter_make_interactive_plugin,
        frontend=InteractiveAdapter_make_interactive_frontend,
        options=None,
        writeback=None,
    ),
    post_analysis=None,
)
