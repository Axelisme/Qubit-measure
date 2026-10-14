from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_analysis_float,
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.mist.power.core import (
    PowerAnalyzeOptions,
    PowerCfg,
    PowerExp,
    PowerResult,
)
from zcu_lab.v2.singleshot.mist.power.form import make_form

MistPowerRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


@dataclass
class MistPowerAnalyzeResult(AnalyzeResultBase):
    pass


@dataclass
class MistPowerAnalyzeParams:
    """MIST plot settings; log_scale uses a logarithmic gain/photon-number axis."""

    log_scale: Annotated[bool, ParamMeta(label="Logarithmic x-axis")] = False


class MistPowerAdapter(
    BaseAdapter[
        PowerCfg, MistPowerRunResult, MistPowerAnalyzeResult, MistPowerAnalyzeParams
    ]
):
    experiment_tag: ClassVar[str | None] = PowerExp.persistence_tag()

    @staticmethod
    def make_form(ctx: FormContext) -> Form[PowerCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = PowerExp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    # No get_analyze_params override: NoAnalyzeParams (4th generic arg).

    def run(
        self,
        req: RunRequest,
        cfg: PowerCfg,
        *,
        context: RunContext,
    ) -> MistPowerRunResult:
        del req
        return RunRecord(cfg, PowerExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistPowerRunResult, MistPowerAnalyzeParams],
        *,
        plots: Plots,
    ) -> MistPowerAnalyzeResult:
        options = PowerAnalyzeOptions(
            ac_coeff=read_analysis_float(req.parameters, "$qubit.ext.ac_stark_coeff"),
            log_scale=req.analyze_params.log_scale,
            confusion_matrix=read_confusion_matrix(req.parameters),
        )
        PowerExp().analyze(req.run_result, options, plots=plots)
        return MistPowerAnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_mist_power_{time.strftime('%m%d')}"
