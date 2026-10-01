from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, TypeAlias, cast

from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.singleshot import GE_Cfg, GE_Exp
from zcu_tools.experiment.v2.singleshot.ge import (
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalyzeOptions,
)
from zcu_tools.experiment.v2_gui.measure.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.measure.adapters.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    MetaDictWriteback,
    ParamMeta,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    PostWritebackRequest,
    RunRequest,
    SessionEnv,
    T_PostAnalyzeResult,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)

if TYPE_CHECKING:
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
        }


@dataclass
class GEPostAnalyzeParams:
    """The GE confusion diagnostic has no independent operator parameters."""


@dataclass(frozen=True)
class GEPostAnalyzeResult(PostAnalyzeResultBase):
    ge_radius: float
    confusion: list[list[float]]


class GEAdapter(BaseAdapter[GE_Cfg, GERunResult, GEAnalyzeResult, GEAnalyzeParams]):
    exp_cls = GE_Exp
    ExpCfg_cls: ClassVar[Any] = GE_Cfg
    # FIT primary analysis + a confusion-diagnostic post-analysis layer.
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot ground/excited readout: measures without and with the "
            "probe pi-pulse, takes 'shots' "
            "single-shot readouts of each, and fits the two IQ clusters to "
            "extract the assignment fidelity, rotation angle and threshold. "
            "Runs on real hardware; the domain forces rounds=1 and reps=shots, "
            "running the readout twice (probe off / on) internally."
        ),
        expects_md=(
            "Reads from the MetaDict (all optional): 't1' — sets the relax "
            "delay as 5*t1 (absent → a fixed 100 us); 'r_f' / 'res_ch' / "
            "'ro_ch' / 'timeFly' / 'best_ro_*' seed the pulse-readout module; "
            "'q_f' / 'qub_ch' seed the probe pi-pulse drive."
        ),
        expects_ml=(
            "Needs a probe pulse (a library pi pulse — 'pi_amp' — when "
            "present) and a pulse-readout module (references a calibrated "
            "library readout 'readout_dpm' / 'readout_rf' when present, else a "
            "blank inline pulse readout). Optionally references a calibrated "
            "reset and an init pulse — both disabled when no library entry "
            "exists."
        ),
        typical_writeback=(
            "Primary proposes the fitted assignment fidelity into MetaDict 'fid', the "
            "cluster width into 'ge_s', and the complex discrimination centres into "
            "'g_center' / 'e_center'. Post-Analysis proposes the optimised "
            "classification radius into 'ge_radius' and the 3x3 confusion matrix "
            "(nested list) into 'confusion_matrix' (a non-scalar, read-only "
            "writeback item)."
        ),
        recommended=(
            "Set Initial State to the predominant state after reset/init and before "
            "the probe pi-pulse. This labels the two acquisitions, not a pure-state prior. "
            "Use a large 'shots' (~1e5) so the IQ histograms are well sampled; "
            "the default analysis backend is 'pca'. Analysis also exposes histogram "
            "log scale, T1 alignment, and an optional shared length ratio; advanced "
            "population priors remain internal. Run once the qubit pi-pulse "
            "and the readout are both calibrated — a clean two-cluster IQ "
            "scatter indicates good discrimination. Use Post-Analysis to inspect "
            "the classified shots and 3x3 confusion diagnostic derived from the "
            "primary fit."
        ),
    )

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .reset(optional=True)
            .pulse("init_pulse", role_id="pi_pulse", optional=True)
            .pulse("probe_pulse", role_id="pi_pulse", label="Probe Pulse")
            .readout(pulse_only=True)
            .relax_delay(scaled_md("t1", factor=5.0, fallback_value=100.0))
            .int("shots", label="Shots", default=100000)
            .reps(1, locked=True)
            .rounds(1, locked=True)
            .build()
        )

    def run(
        self, req: RunRequest, raw_cfg: dict[str, object], *, plots: Plots
    ) -> GERunResult:
        soc, soccfg = require_soc_handles(req)
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = GE_Exp().run(cfg, context=QickContext(soc, soccfg, plots))
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
        # Float scalars plus the complex discrimination centres. complex md
        # values round-trip end-to-end now (in-process apply + MetaDict str
        # persistence both speak complex; the wire carries {"__complex__": [...]}
        # and the UI parses "re+imj"). Mirrors the notebook's md.g_center /
        # md.e_center.
        return [
            MetaDictWriteback(
                target_name="fid",
                description="Single-shot assignment fidelity",
                proposed_value=result.fidelity,
            ),
            MetaDictWriteback(
                target_name="ge_s",
                description="Single-shot IQ cluster width (s)",
                proposed_value=result.ge_s,
            ),
            MetaDictWriteback(
                target_name="g_center",
                description="Single-shot |g> IQ cluster centre (complex)",
                proposed_value=result.g_center,
            ),
            MetaDictWriteback(
                target_name="e_center",
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
            MetaDictWriteback(
                target_name="ge_radius",
                description="Single-shot classification radius",
                proposed_value=result.ge_radius,
            ),
            MetaDictWriteback(
                target_name="confusion_matrix",
                description="Single-shot 3x3 confusion matrix (prepared->measured)",
                proposed_value=result.confusion,
            ),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_sh_ge_{time.strftime('%m%d')}"
