from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterGuide,
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    SampleWriteback,
    SessionEnv,
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

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "T1 energy relaxation: applies a pi pulse to excite the qubit, then sweeps "
            "a wait delay before readout and fits the exponential decay to extract T1. "
            "Runs on real hardware. Run after a pi pulse has been calibrated "
            "(amplitude/length Rabi)."
        ),
        expects_md=(
            "Delay runs 0 to 5 * qubit.t1 us over 101 points. Relax delay is 5 * "
            "qubit.t1 us. Missing calibration leaves the affected fields empty and "
            "Invalid; no numeric fallback."
        ),
        expects_ml=(
            "Uses captured qubit.module.readout with a readout template fallback, and "
            "qubit.module.reset with no fallback (disabled when absent). The drive uses "
            "qubit.module.x180 with a pi_pulse template fallback. Unset or dangling "
            "slots use the named template, not a legacy blank. Key choices remain "
            "available. Inline templates read qubit.freq, qubit.wiring.L01.ch, "
            "resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + "
            "0.05 us; missing required calibration remains Invalid."
        ),
        typical_writeback=(
            "Proposes fitted qubit.t1 and its standard error (us) to the captured qubit "
            "and working point. Also offers an independent sample with t1_us/t1_err_us "
            "at the Run bias when captured device evidence is available. No module "
            "writeback."
        ),
        recommended=(
            "Analysis defaults to a single-exponential fit; enable dual-exponential "
            "only when the decay clearly shows two timescales (a fast component on top "
            "of the slow relaxation). A delay sweep reaching ~5*T1 lets the decay "
            "flatten so the fit is well-constrained. Supply a delay endpoint when "
            "qubit.t1 is missing. Set 'uniform=False' to cluster more points on the "
            "exponential decay while keeping the configured start/stop window."
        ),
    )

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

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_t1_{time.strftime('%m%d')}"
