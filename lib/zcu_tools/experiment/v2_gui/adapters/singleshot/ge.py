from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from zcu_tools.experiment.utils.single_shot import singleshot_ge_analysis
from zcu_tools.experiment.v2.singleshot import GE_Cfg, GE_Exp
from zcu_tools.experiment.v2.singleshot.ge import GE_Result
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.shared import (
    CfgBuilder,
    Init,
    build_exp_spec,
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_reset_module_spec,
    proper_relax,
)
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    IntSpec,
    LiteralSpec,
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
    # ``backend`` selects the rotation/threshold method. This phase ships the
    # primary analysis fixed to PCA (the domain default); the other backends are
    # the post-analysis (multi-method) phase, so only the two implemented choices
    # are offered (the Literal supplies the form's choices) and the default is
    # "pca".
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
    ge_radius: float
    confusion: list[list[float]]
    figure: Figure


@dataclass
class GEPostAnalyzeParams:
    # The post-analysis (multi-method) layer: re-runs the discrimination with a
    # user-chosen ``backend``, or — when ``angle`` is supplied — a manual rotation
    # (``angle`` overrides ``backend`` in the domain fitter). ``regression`` is
    # intentionally NOT offered here (it is excluded from this adapter's surface).
    backend: Annotated[Literal["pca", "center"], ParamMeta(label="Backend")] = "pca"
    # ``angle`` (radians): when set, the domain ignores ``backend`` and rotates by
    # this fixed angle (manual discrimination). Optional → blank means "use
    # backend".
    angle: Annotated[float | None, ParamMeta(label="Manual angle (rad)")] = None


@dataclass
class GEPostAnalyzeResult(PostAnalyzeResultBase):
    # Same float scalars as the primary result (JSON-safe via to_summary_dict).
    # ``g_center`` / ``e_center`` are complex and auto-skipped from the summary.
    backend: str
    fidelity: float
    theta: float
    threshold: float
    ge_s: float
    g_center: complex
    e_center: complex
    figure: Figure


class GEAdapter(BaseAdapter[GE_Cfg, GERunResult, GEAnalyzeResult, GEAnalyzeParams]):
    exp_cls = GE_Exp
    ExpCfg_cls: ClassVar[Any] = GE_Cfg
    # FIT primary analysis + opt-in post-analysis (the multi-backend
    # discrimination layer).
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True
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
            "scatter indicates good discrimination."
        ),
    )

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return build_exp_spec(
            # Module field order mirrors GEModuleCfg: reset, init_pulse,
            # probe_pulse, readout.
            modules={
                "reset": make_reset_module_spec(optional=True),
                "init_pulse": make_pulse_module_spec(optional=True),
                "probe_pulse": make_pulse_module_spec(label="Probe Pulse"),
                "readout": make_pulse_readout_module_spec(),
            },
            # Single-shot has no swept axis; the shot count is the run-only knob
            # that the domain copies into reps.
            extra={"shots": IntSpec(label="Shots")},
            # The domain overwrites reps (← shots) and rounds (← 1) at run, so lock
            # them off the form (lookback locks reps the same way) — showing fields
            # the run silently discards would be misleading.
            reps=LiteralSpec(value=1, label="Reps"),
            rounds=LiteralSpec(value=1, label="Rounds"),
        )

    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        return (
            CfgBuilder(ctx, self.cfg_spec())
            .scalars(shots=100000)
            .set("relax_delay", proper_relax(ctx))
            .role("modules.probe_pulse", "pi_pulse", Init.INLINE)
            .role("modules.readout", "readout")
            # optional → None (disabled) when no library entry (ADR-0010)
            .role("modules.reset", "reset", Init.DISABLED)
            .role("modules.init_pulse", "pi_pulse", Init.DISABLED)
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
        # ``pops`` (the fit's 2×2 [[p0_gg, p0_ge], [p0_eg, p0_ee]]) is the
        # ``init_pops`` the confusion calc needs — fully derived from the primary
        # fit, so no extra analyze parameter. ``radius=None`` lets the domain
        # optimise ``ge_radius``. ``consider_other=False`` mirrors the notebook
        # single-shot flow. The confusion figure is discarded here: the result
        # displays the primary fit figure, and the matrix is shown via the JSON
        # summary (closing it avoids leaking an open Figure).
        confusion, ge_radius, confusion_fig = exp.calc_confusion_matrix(
            pops,
            g_center,
            e_center,
            radius=None,
            result=req.run_result,
            consider_other=False,
        )
        plt.close(confusion_fig)
        return GEAnalyzeResult(
            fidelity=fidelity,
            theta=fit_result["theta"],
            threshold=fit_result["threshold"],
            ge_s=fit_result["s"],
            g_center=g_center,
            e_center=e_center,
            ge_radius=ge_radius,
            confusion=confusion.tolist(),
            figure=fig,
        )

    def get_post_analyze_params(
        self, analyze_result: GEAnalyzeResult, ctx: ExpContext
    ) -> GEPostAnalyzeParams:
        del analyze_result, ctx
        # Default the post-analysis to the same backend the primary uses (pca),
        # no manual angle.
        return GEPostAnalyzeParams(backend="pca", angle=None)

    def post_analyze(
        self,
        req: PostAnalyzeRequest[GERunResult, GEAnalyzeResult, GEPostAnalyzeParams],
    ) -> GEPostAnalyzeResult:
        params = req.post_analyze_params
        # ``singleshot_ge_analysis`` ignores ``backend`` when ``angle`` is given
        # (manual rotation), so pass both through verbatim — the domain owns the
        # precedence. ``effective_backend`` records which path actually ran.
        fidelity, _pops, fit_result, fig = singleshot_ge_analysis(
            req.run_result.signals,
            angle=params.angle,
            backend=params.backend,
        )
        effective_backend = "manual" if params.angle is not None else params.backend
        return GEPostAnalyzeResult(
            backend=effective_backend,
            fidelity=fidelity,
            theta=fit_result["theta"],
            threshold=fit_result["threshold"],
            ge_s=fit_result["s"],
            g_center=fit_result["g_center"],
            e_center=fit_result["e_center"],
            figure=fig,
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
