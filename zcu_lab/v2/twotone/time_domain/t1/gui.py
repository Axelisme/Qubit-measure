from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SampleWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2.twotone.time_domain.t1.core import (
    T1Analysis,
    T1AnalyzeOptions,
    T1Cfg,
    T1Exp,
    T1Result,
)

from .form import make_form

if TYPE_CHECKING:
    from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

T1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


@dataclass
class T1AnalyzeParams:
    dual_exp: Annotated[bool, ParamMeta(label="Dual exponential")] = False
    skip: Annotated[int, ParamMeta(label="Skip leading points")] = 0


@dataclass(frozen=True)
class T1AnalyzeResult:
    """GUI summary over the unchanged, typed core analysis."""

    analysis: T1Analysis

    @property
    def t1(self) -> float:
        return self.analysis.t1

    @property
    def t1_err(self) -> float:
        return self.analysis.t1_err

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "t1": self.analysis.t1,
            "t1_err": self.analysis.t1_err,
            "t1b": self.analysis.t1b,
            "t1b_err": self.analysis.t1b_err,
            "fit_quality": fit_quality_summary(self.analysis.fit_quality),
        }


class T1Adapter(BaseAdapter[T1Cfg, T1RunResult, T1AnalyzeResult, T1AnalyzeParams]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[T1Cfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = T1Exp
    experiment_tag: ClassVar[str | None] = T1Exp.persistence_tag()

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: T1Cfg,
        *,
        context: RunContext,
    ) -> T1RunResult:
        del req
        result = T1Exp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[T1RunResult, T1AnalyzeParams], *, plots: Plots
    ) -> T1AnalyzeResult:
        params = req.analyze_params
        if params.skip < 0:
            raise ValueError("Skip leading points must be nonnegative")
        analysis = T1Exp().analyze(
            req.run_result,
            T1AnalyzeOptions(dual_exp=params.dual_exp, skip=params.skip),
            plots=plots,
        )
        return T1AnalyzeResult(analysis)

    def get_writeback_items(
        self, req: WritebackRequest[T1RunResult, T1AnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose fitted T1 parameters and a captured-component sample.

        req supplies the source result and fixed Run context. Sample fields use
        us and target the Run's resolved $qubit, never the live selection. Without
        a parameter capture, retain parameter proposals only so loaded data can
        still be analyzed; Apply rejects the missing capture. Unbound $qubit
        with a capture raises RoleResolutionError. WritebackService supplies
        bias coordinates or omits the sample with its existing public reason.
        No write or device access occurs here; this Run leaves devices fixed.
        """
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.t1",
                description="T1 relaxation time (us)",
                proposed_value=result.t1,
                stderr=result.t1_err,
            ),
        ]
        parameters = req.ctx.parameters
        if parameters is not None:
            items.append(
                SampleWriteback(
                    target_name=parameters.resolve_path("$qubit"),
                    description="T1 sample at the captured bias",
                    proposed_value={
                        "t1_us": result.t1,
                        "t1_err_us": result.t1_err,
                    },
                )
            )
        return items
