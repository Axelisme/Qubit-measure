from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, TypeAlias, cast

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    PostWritebackRequest,
    RunRequest,
    SessionEnv,
    T_PostAnalyzeResult,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2.singleshot.ge.core import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalyzeOptions,
)
from zcu_lab.v2.singleshot.ge.form import make_form

if TYPE_CHECKING:
    from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

GERunResult: TypeAlias = RunRecord[GE_Cfg, GE_Result]


@dataclass
class GEAnalyzeParams:
    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="Initial State")
    ] = "ground"
    # ``backend`` selects the primary rotation/threshold fit. Post-analysis uses
    # the resulting centres and does not choose or run another fit backend.
    backend: Annotated[Literal["pca", "center"], ParamMeta(label="Backend")] = "pca"
    logscale: Annotated[bool, ParamMeta(label="Log Scale")] = False
    align_t1: Annotated[bool, ParamMeta(label="Align T1")] = True
    length_ratio: Annotated[
        float | None, ParamMeta(label="Length Ratio", decimals=4)
    ] = None


@dataclass(frozen=True)
class GEAnalyzeResult(GEAnalysis, AnalyzeResultBase):
    """Fit calibration, with a JSON-safe GUI projection of the core result."""

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "initial_state": self.initial_state,
            "fidelity": self.fidelity,
            "theta": self.theta,
            "threshold": self.threshold,
            "ge_s": self.ge_s,
            "init_pops": self.init_pops.tolist(),
            "fit_quality": fit_quality_summary(self.fit_quality),
        }


@dataclass
class GEPostAnalyzeParams:
    """The GE confusion diagnostic has no independent operator parameters."""


@dataclass(frozen=True)
class GEPostAnalyzeResult(PostAnalyzeResultBase):
    ge_radius: float
    confusion: list[list[float]]


class GEAdapter(BaseAdapter[GE_Cfg, GERunResult, GEAnalyzeResult, GEAnalyzeParams]):
    experiment_tag: ClassVar[str | None] = GE_Exp.persistence_tag()

    @staticmethod
    def make_form(ctx: FormContext) -> Form[GE_Cfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = GE_Exp
    # FIT primary analysis + a confusion-diagnostic post-analysis layer.
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot ground/excited readout: measures without and with the probe "
            "pi-pulse, takes 'shots' single-shot readouts of each, and fits the two IQ "
            "clusters to extract the assignment fidelity, rotation angle and threshold. "
            "Runs on real hardware; the domain forces rounds=1 and reps=shots, running "
            "the readout twice (probe off / on) internally."
        ),
        expects_md=(
            "Open and Reset capture the current aliases, parameters and library. Relax "
            "delay is 5 * $qubit.t1. Inline probe uses $qubit.freq and "
            "$qubit.wiring.L01.ch. Inline templates read $qubit.freq, "
            "$qubit.wiring.L01.ch and $resonator.freq/wiring.ch/wiring.ro_ch; readout "
            "trigger uses $resonator.ext.time_of_fly + 0.05 us. Missing required sources "
            "remain Invalid; no numeric fallback is supplied."
        ),
        expects_ml=(
            "Uses $qubit.module.x180 for optional init and required probe, "
            "$qubit.module.reset for optional reset, and $qubit.module.readout for "
            "readout. Missing required slots use pi_pulse/readout templates; missing "
            "optional slots stay disabled. Run uses this fixed capture."
        ),
        typical_writeback=(
            "Writes to the captured qubit.ext.readout_cal: primary proposes fidelity, "
            "sigma, g_center and e_center; post-analysis proposes radius and the "
            "nested-list confusion_matrix. Centers remain native complex proposals; "
            "accepted values use the shared __complex__ tag. Changing point or focus does "
            "not retarget an existing proposal."
        ),
        recommended=(
            "Set Initial State to the predominant state after reset/init and before the "
            "probe pi-pulse. This labels the two acquisitions, not a pure-state prior. "
            "Use a large 'shots' (~1e5) so the IQ histograms are well sampled; the "
            "default analysis backend is 'pca'. Analysis also exposes histogram log "
            "scale, T1 alignment, and an optional shared length ratio; advanced "
            "population priors remain internal. Run once the qubit pi-pulse and the "
            "readout are both calibrated — a clean two-cluster IQ scatter indicates good "
            "discrimination. Use Post-Analysis to inspect the classified shots and 3x3 "
            "confusion diagnostic derived from the primary fit."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: GE_Cfg,
        *,
        context: RunContext,
    ) -> GERunResult:
        del req
        result = GE_Exp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[GERunResult, GEAnalyzeParams], *, plots: Plots
    ) -> GEAnalyzeResult:
        params = req.analyze_params
        analysis = GE_Exp().analyze(
            req.run_result,
            GEAnalyzeOptions(
                initial_state=params.initial_state,
                backend=params.backend,
                logscale=params.logscale,
                align_t1=params.align_t1,
                length_ratio=params.length_ratio,
            ),
            plots=plots,
        )
        return GEAnalyzeResult(
            initial_state=analysis.initial_state,
            fidelity=analysis.fidelity,
            theta=analysis.theta,
            threshold=analysis.threshold,
            ge_s=analysis.ge_s,
            g_center=analysis.g_center,
            e_center=analysis.e_center,
            init_pops=analysis.init_pops,
            fit_quality=analysis.fit_quality,
        )

    def get_post_analyze_params(
        self, analyze_result: GEAnalyzeResult, ctx: SessionEnv
    ) -> GEPostAnalyzeParams:
        del analyze_result, ctx
        return GEPostAnalyzeParams()

    def post_analyze(
        self,
        req: PostAnalyzeRequest[GERunResult, GEAnalyzeResult, GEPostAnalyzeParams],
        *,
        plots: Plots,
    ) -> GEPostAnalyzeResult:
        analysis = GE_Exp().post_analyze(
            req.run_result, req.analyze_result, GEPostAnalyzeOptions(), plots=plots
        )
        return GEPostAnalyzeResult(
            ge_radius=analysis.confusion.radius,
            confusion=analysis.confusion.matrix.tolist(),
        )

    def get_writeback_items(
        self, req: WritebackRequest[GERunResult, GEAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        result.validate_calibration()
        # AcceptedWrite owns native complex conversion; producers keep numbers.
        return [
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.fidelity",
                description="Single-shot assignment fidelity",
                proposed_value=result.fidelity,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.sigma",
                description="Single-shot IQ cluster width (s)",
                proposed_value=result.ge_s,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.g_center",
                description="Single-shot |g> IQ cluster centre (complex)",
                proposed_value=result.g_center,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.e_center",
                description="Single-shot |e> IQ cluster centre (complex)",
                proposed_value=result.e_center,
            ),
        ]

    def get_post_writeback_items(
        self,
        req: PostWritebackRequest[GERunResult, GEAnalyzeResult, T_PostAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        result = cast(GEPostAnalyzeResult, req.post_analyze_result)
        return [
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.radius",
                description="Single-shot classification radius",
                proposed_value=result.ge_radius,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.readout_cal.confusion_matrix",
                description="Single-shot 3x3 confusion matrix (prepared->measured)",
                proposed_value=result.confusion,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sh_ge_{time.strftime('%m%d')}"
