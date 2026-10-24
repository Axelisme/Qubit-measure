"""Measure GUI adapter for the fast-flux MIST contrast map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    ParamMeta,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.fastflux.mist.core import (
    MistAnalyzeOptions,
    MistCfg,
    MistExp,
    MistResult,
)
from zcu_lab.v2.fastflux.mist.form import make_form

MistRunResult: TypeAlias = RunRecord[MistCfg, MistResult]


@dataclass
class MistAnalyzeParams:
    ac_coeff: Annotated[
        float | None, ParamMeta(label="Photon number / gain squared")
    ] = None


class MistAdapter(BaseAdapter[MistCfg, MistRunResult, None, MistAnalyzeParams]):
    exp_cls = MistExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MistCfg]:
        """Build editable settings from the captured Open/Reset sources."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: MistCfg, *, context: RunContext
    ) -> MistRunResult:
        """Acquire the gain map and retain its exact source settings."""
        del req
        return RunRecord(cfg, MistExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistRunResult, MistAnalyzeParams],
        *,
        plots: Plots,
    ) -> None:
        """Plot the stored map, optionally scaling gain squared to photons."""
        MistExp().analyze(
            req.run_result,
            MistAnalyzeOptions(ac_coeff=req.analyze_params.ac_coeff),
            plots=plots,
        )
        return
