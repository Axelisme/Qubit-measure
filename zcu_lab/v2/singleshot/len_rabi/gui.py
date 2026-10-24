from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Literal, TypeAlias

from zcu_tools.analysis.fitting.singleshot_rabi import RabiJointFitResult
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_rabi import rabi_calibration_writeback
from zcu_lab.v2.singleshot.len_rabi.core import (
    LenRabiAnalyzeOptions,
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)
from zcu_lab.v2.singleshot.len_rabi.form import make_form

# ``LenRabiExp`` from ``singleshot`` — sweeps the qubit-drive pulse *length* and
# preserves every raw IQ shot. Analysis derives populations from that canonical
# raw result rather than persisting a second population representation.
SsLenRabiRunResult: TypeAlias = RunRecord[LenRabiCfg, LenRabiResult]


@dataclass
class SsLenRabiAnalyzeParams:
    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="Initial State")
    ] = "ground"
    decay: Annotated[bool, ParamMeta(label="Fit decay envelope")] = True
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset")] = False


@dataclass
class SsLenRabiAnalyzeResult(AnalyzeResultBase):
    # The full numeric fit is intentionally non-JSON-safe and therefore omitted
    # from the GUI summary. The operator reviews the population/fit Figure while
    # writeback projection reads the typed domain result directly.
    fit_result: RabiJointFitResult


class SsLenRabiAdapter(
    BaseAdapter[
        LenRabiCfg, SsLenRabiRunResult, SsLenRabiAnalyzeResult, SsLenRabiAnalyzeParams
    ]
):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[LenRabiCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = LenRabiExp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: LenRabiCfg,
        *,
        context: RunContext,
    ) -> SsLenRabiRunResult:
        del req
        result = LenRabiExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[SsLenRabiRunResult, SsLenRabiAnalyzeParams],
        *,
        plots: Plots,
    ) -> SsLenRabiAnalyzeResult:
        fit_result = LenRabiExp().analyze(
            req.run_result,
            LenRabiAnalyzeOptions(
                decay=req.analyze_params.decay,
                fit_phase=req.analyze_params.fit_phase,
                initial_state=req.analyze_params.initial_state,
            ),
            plots=plots,
        )
        return SsLenRabiAnalyzeResult(fit_result=fit_result)

    def get_writeback_items(
        self, req: WritebackRequest[SsLenRabiRunResult, SsLenRabiAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return rabi_calibration_writeback(req.analyze_result.fit_result)
