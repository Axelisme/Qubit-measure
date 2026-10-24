from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    ParameterWriteback,
    RunRequest,
    SampleWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter

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


class T1Summary(BaseModel):
    """Recipe JSON inputs in native units; null means an unavailable estimate."""

    t1: float | None = Field(
        description="Owner t1 in us; null retains an unavailable estimate."
    )

    t1_err: float | None = Field(
        description="Owner t1_err in us; null retains an unavailable estimate."
    )

    t1b: float | None = Field(
        description="Owner t1b in us; null retains an unavailable estimate."
    )

    t1b_err: float | None = Field(
        description="Owner t1b_err in us; null retains an unavailable estimate."
    )


class T1Adapter(BaseAdapter[T1Cfg, T1RunResult, T1Analysis, T1AnalyzeOptions]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[T1Cfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = T1Exp
    sample_device_alias: ClassVar[str | None] = "$qubit.flux_source"

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
        self, req: AnalyzeRequest[T1RunResult, T1AnalyzeOptions], *, plots: Plots
    ) -> T1Analysis:
        params = req.analyze_params
        if params.skip < 0:
            raise ValueError("Skip leading points must be nonnegative")
        return T1Exp().analyze(
            req.run_result,
            req.analyze_params,
            plots=plots,
        )

    def get_writeback_items(
        self, req: WritebackRequest[T1RunResult, T1Analysis]
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
