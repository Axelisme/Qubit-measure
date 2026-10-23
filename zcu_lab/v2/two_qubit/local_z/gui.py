"""Measure adapter for a compound, source-bound local-Z correction pair."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    NoAnalyzeParams,
    ParameterWriteback,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .analysis import LocalZAnalysis
from .core import LocalZCfg, LocalZExp, LocalZResult
from .form import make_form

LocalZRun: TypeAlias = RunRecord[LocalZCfg, LocalZResult]


@dataclass(frozen=True)
class LocalZAnalyzeResult:
    """JSON-safe phase and linear-fit summary; figures remain named artifacts."""

    analysis: LocalZAnalysis

    def to_summary_dict(self) -> dict[str, object]:
        """Expose both targets' phases, fit diagnostics and missing-value reasons."""
        return {
            "targets": [
                {
                    "target": role,
                    "points": [
                        {
                            "gate_count": n,
                            "contrast": p.contrast if np.isfinite(p.contrast) else None,
                            "logical_phase_deg": p.logical_phase_deg,
                            "reason": p.reason,
                        }
                        for n, p in zip(
                            self.analysis.gate_counts,
                            self.analysis.phases[t],
                            strict=True,
                        )
                    ],
                    "intercept_deg": fit.intercept_deg,
                    "slope_deg": fit.slope_deg,
                    "correction_deg": fit.correction_deg,
                    "slope_error_deg": fit.slope_error_deg,
                    "reason": fit.reason,
                    "fit_quality": None
                    if fit.fit_quality is None
                    else fit.fit_quality.to_summary_dict(),
                }
                for t, (role, fit) in enumerate(
                    zip(("f1", "f2"), self.analysis.fits, strict=True)
                )
            ],
            "acquired_shots": self.analysis.acquired_shots.tolist(),
            "model": "Complete-row adjacent principal-value unwrap; OLS phase_deg = intercept + slope*n. correction=-slope, not rewrapped. Covariance uses residual variance, not a shot-noise model. No automatic statistical gate.",
            "interpretation": "B00 thermal ensemble, other qubit not flipped. Assumes adjacent true phase increments have magnitude <180 deg; extra 360 deg per gate is unidentifiable. Linear mixture estimate, not purified local phase or fidelity. No heralding or shot exclusion.",
        }


class LocalZAdapter(
    BaseAdapter[LocalZCfg, LocalZRun, LocalZAnalyzeResult, NoAnalyzeParams]
):
    """Fixed-capture Form, both-target Run, saved-source analysis and native restore."""

    exp_cls = LocalZExp
    experiment_tag: ClassVar[str | None] = LocalZExp.persistence_tag()
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[LocalZCfg]:
        """Capture roles, complete mux, X90s, CZ pulses and GE calibration on Open/Reset."""
        return make_form(ctx)

    def run(self, req: RunRequest, cfg: LocalZCfg, *, context: RunContext) -> LocalZRun:
        """Keep original cfg and full raw; caller checks cancel_signal.raise_if_error()."""
        del req
        return RunRecord(cfg=cfg, result=LocalZExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[LocalZRun, NoAnalyzeParams], *, plots: Plots
    ) -> LocalZAnalyzeResult:
        """Fit only saved cfg/raw, never current Form, aliases or library."""
        return LocalZAnalyzeResult(
            LocalZExp().analyze(req.run_result, None, plots=plots)
        )

    def get_writeback_items(
        self, req: WritebackRequest[LocalZRun, LocalZAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose one component-named pair only when both target fits are valid.

        Saved cfg and Run/Load parameter capture are required. Raise ValueError
        if its F1/F2/coupler identities differ from the saved source, including
        swapped roles; never silently reuse a pair. Preview/selected Apply is
        manual. No pulse, constructive relative phase or other parameter changes.
        """
        cfg, parameters = req.run_result.cfg, req.ctx.parameters
        if cfg is None or parameters is None:
            return []
        for role, name in (
            ("f1", cfg.roles.f1),
            ("f2", cfg.roles.f2),
            ("coupler", cfg.roles.coupler),
        ):
            if parameters.resolve_path(f"${role}") != name:
                raise ValueError(
                    "Local-Z source F1/F2/coupler identities differ from writeback capture; Reset and remeasure for changed roles"
                )
        first, second = req.analyze_result.analysis.fits
        if first.correction_deg is None or second.correction_deg is None:
            return []
        return [
            ParameterWriteback(
                target_name="$coupler.ext.cz_phase",
                description="B00 thermal linear-model logical corrections (deg); source F1/F2 identities, correction = −slope",
                proposed_value={
                    "f1": cfg.roles.f1,
                    "f2": cfg.roles.f2,
                    "correction_f1_deg": first.correction_deg,
                    "correction_f2_deg": second.correction_deg,
                },
            )
        ]
