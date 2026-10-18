"""Fixed-capture measure attachment for common-amplitude CZ return scans."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    NoAnalyzeParams,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .analysis import CZReturnData, classify_return, plot_return
from .core import CZAmplitudeCfg, CZAmplitudeExp, CZAmplitudeResult
from .form import make_form

CZAmplitudeRun: TypeAlias = RunRecord[CZAmplitudeCfg, CZAmplitudeResult]


@dataclass(frozen=True)
class CZAmplitudeAnalyzeResult:
    """JSON-safe complete-scale shot counts and joint-return diagnostics."""

    data: CZReturnData

    def to_summary_dict(self) -> dict[str, object]:
        """Report incomplete scales without substituting zero probabilities."""
        return {
            "analysis_kind": "classification_only",
            "complete_scales": int(self.data.complete.sum()),
            "acquired_shots": self.data.acquired_shots.tolist(),
            "analyzed_shots": self.data.analyzed_shots.tolist(),
            "interpretation": "Thermal nominal B11; same-shot return is a leakage proxy, not physical gate fidelity. No heralding, radius cuts or confusion inversion.",
        }


class CZAmplitudeAdapter(
    BaseAdapter[
        CZAmplitudeCfg, CZAmplitudeRun, CZAmplitudeAnalyzeResult, NoAnalyzeParams
    ]
):
    """Acquisition checkpoint with classification diagnostics, no amplitude estimator.

    FIT invokes saved-source nearest-center classification and IQ plots only.
    No fitted scale, calibration proposal or writeback is supplied by this adapter.
    """

    exp_cls = CZAmplitudeExp
    experiment_tag: ClassVar[str | None] = CZAmplitudeExp.persistence_tag()
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[CZAmplitudeCfg]:
        """Resolve roles/tone identities and editable calibrations once at Open/Reset."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: CZAmplitudeCfg, *, context: RunContext
    ) -> CZAmplitudeRun:
        """Pair unchanged source cfg and raw; caller checks cancel_signal.raise_if_error()."""
        del req
        return RunRecord(cfg=cfg, result=CZAmplitudeExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[CZAmplitudeRun, NoAnalyzeParams], *, plots: Plots
    ) -> CZAmplitudeAnalyzeResult:
        """Classify only complete scales using saved cfg, never current Form/library."""
        answer = classify_return(req.run_result)
        plot_return(req.run_result, answer, plots=plots)
        return CZAmplitudeAnalyzeResult(answer)
