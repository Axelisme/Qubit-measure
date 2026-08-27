from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import numpy as np
from matplotlib.figure import Figure

from zcu_tools.experiment.v2.singleshot import GE_Cfg, GE_Exp
from zcu_tools.experiment.v2.singleshot.ge import GE_Result
from zcu_tools.experiment.v2_gui.adapters._support import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
    scaled_md,
)
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    ExpContext,
    MetaDictWriteback,
    ParamMeta,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    WritebackItem,
    WritebackRequest,
)

GERunResult: TypeAlias = GE_Result


@dataclass
class GEAnalyzeParams:
    # ``backend`` selects the primary rotation/threshold fit. Post-analysis uses
    # the resulting centres and does not choose or run another fit backend.
    backend: Annotated[Literal["pca", "center"], ParamMeta(label="Backend")] = "pca"


@dataclass
class GEAnalyzeResult(AnalyzeResultBase):
    # ``fidelity`` and ``ge_s`` are plain floats (writeback-safe). ``g_center`` /
    # ``e_center`` are complex — kept here for downstream post-analysis use, but
    # skipped from ``to_summary_dict`` automatically (complex is not JSON-safe).
    fidelity: float
    theta: float
    threshold: float
    ge_s: float
    g_center: complex
    e_center: complex
    # ``ge_radius`` is the optimised classification radius (writeback-safe float;
    # the per-qubit calibration downstream single-shot experiments consume).
    # ``confusion`` is the 3×3 prepared→measured confusion matrix as a nested
    # ``list[list[float]]`` so ``to_summary_dict`` carries it JSON-safe (the
    # domain returns a numpy array). Both come from
    # ``GE_Exp.calc_confusion_matrix`` over the primary fit's populations.
    init_pops: list[list[float]]
    ge_radius: float
    confusion: list[list[float]]
    figure: Figure


@dataclass
class GEPostAnalyzeParams:
    """The GE confusion diagnostic has no independent operator parameters."""


@dataclass
class GEPostAnalyzeResult(PostAnalyzeResultBase):
    ge_radius: float
    confusion: list[list[float]]
    figure: Figure


class GEAdapter(BaseAdapter[GE_Cfg, GERunResult, GEAnalyzeResult, GEAnalyzeParams]):
    exp_cls = GE_Exp
    ExpCfg_cls: ClassVar[Any] = GE_Cfg
    # FIT primary analysis + a confusion-diagnostic post-analysis layer.
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Single-shot ground/excited readout: prepares the qubit in |g> "
            "(no probe pulse) and |e> (probe pi-pulse), takes 'shots' "
            "single-shot readouts of each, and fits the two IQ clusters to "
            "extract the assignment fidelity, rotation angle and threshold. "
            "Runs on real hardware; the domain forces rounds=1 and reps=shots, "
            "running the readout twice (g-prep / e-prep) internally."
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
            "Proposes the fitted assignment fidelity into MetaDict 'fid', the "
            "cluster width into 'ge_s', the complex discrimination centres into "
            "'g_center' / 'e_center', the optimised classification radius into "
            "'ge_radius', and the 3x3 confusion matrix (nested list) into "
            "'confusion_matrix' (a non-scalar, read-only writeback item)."
        ),
        recommended=(
            "Use a large 'shots' (~1e5) so the IQ histograms are well sampled; "
            "the default analysis backend is 'pca'. Run once the qubit pi-pulse "
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

    def analyze(
        self, req: AnalyzeRequest[GERunResult, GEAnalyzeParams]
    ) -> GEAnalyzeResult:
        params = req.analyze_params
        exp = GE_Exp()
        fidelity, pops, fit_result, fig = exp.analyze(
            req.run_result, backend=params.backend
        )
        g_center = fit_result["g_center"]
        e_center = fit_result["e_center"]
        # ``pops`` has fixed order [[p0_gg, p0_ge], [p0_eg, p0_ee]]. The
        # figure-free confusion calculation keeps the primary distribution figure
        # visible while preserving the existing complete writeback.
        ge_s = fit_result["s"]
        confusion = exp.calc_confusion_matrix(
            pops,
            g_center,
            e_center,
            ge_s,
            radius=None,
            result=req.run_result,
            consider_other=False,
        )
        return GEAnalyzeResult(
            fidelity=fidelity,
            theta=fit_result["theta"],
            threshold=fit_result["threshold"],
            ge_s=ge_s,
            g_center=g_center,
            e_center=e_center,
            init_pops=pops.tolist(),
            ge_radius=confusion.radius,
            confusion=confusion.matrix.tolist(),
            figure=fig,
        )

    def get_post_analyze_params(
        self, analyze_result: GEAnalyzeResult, ctx: ExpContext
    ) -> GEPostAnalyzeParams:
        del analyze_result, ctx
        return GEPostAnalyzeParams()

    def post_analyze(
        self,
        req: PostAnalyzeRequest[GERunResult, GEAnalyzeResult, GEPostAnalyzeParams],
    ) -> GEPostAnalyzeResult:
        primary = req.analyze_result
        exp = GE_Exp()
        confusion = exp.calc_confusion_matrix(
            np.asarray(primary.init_pops, dtype=np.float64),
            primary.g_center,
            primary.e_center,
            primary.ge_s,
            radius=None,
            result=req.run_result,
            consider_other=False,
        )
        figure = exp.plot_confusion_matrix(
            confusion,
            primary.g_center,
            primary.e_center,
            result=req.run_result,
        )
        return GEPostAnalyzeResult(
            ge_radius=confusion.radius,
            confusion=confusion.matrix.tolist(),
            figure=figure,
        )

    def get_writeback_items(
        self, req: WritebackRequest[GERunResult, GEAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
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
            # ``ge_radius`` is the per-qubit classification radius downstream
            # single-shot experiments consume — a clean scalar, mirrors the
            # notebook's md.ge_radius.
            MetaDictWriteback(
                target_name="ge_radius",
                description="Single-shot classification radius",
                proposed_value=result.ge_radius,
            ),
            # ``confusion_matrix`` is the 3×3 prepared→measured confusion matrix
            # as a nested ``list[list[float]]`` (md key mirrors the notebook's
            # md.confusion_matrix). It is a non-scalar md value: MetaDict already
            # stores nested lists (it cannot hold ndarray — dumps tolist(), loads
            # raw), the value is JSON-safe so the wire carries it as-is, and the
            # writeback UI renders it read-only (derived value, applied verbatim).
            MetaDictWriteback(
                target_name="confusion_matrix",
                description="Single-shot 3x3 confusion matrix (prepared->measured)",
                proposed_value=result.confusion,
            ),
        ]

    def make_filename_stem(self, ctx: ExpContext) -> str:
        return f"{ctx.qub_name}_sh_ge_{time.strftime('%m%d')}"
