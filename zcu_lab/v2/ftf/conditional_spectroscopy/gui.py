"""Measure adapter for four thermal branches of two-line spectroscopy."""

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

from .analysis import SpectroscopyAnalysis
from .core import BRANCH_LABELS, SpectroscopyCfg, SpectroscopyExp, SpectroscopyResult
from .form import make_form

SpectroscopyRun: TypeAlias = RunRecord[SpectroscopyCfg, SpectroscopyResult]


@dataclass(frozen=True)
class SpectroscopyAnalyzeResult:
    """JSON-safe completion summary; diagnostic spectra remain named figures."""

    analysis: SpectroscopyAnalysis

    def to_summary_dict(self) -> dict[str, object]:
        """Report all DC/branch shot counts, never label incomplete means as valid."""
        rows: list[dict[str, object]] = []
        for d in range(self.analysis.complete.shape[0]):
            for b, label in enumerate(BRANCH_LABELS):
                complete = bool(self.analysis.complete[d, b])
                acquired = int(self.analysis.acquired_shots[d, b])
                rows.append(
                    {
                        "dc_index": d,
                        "branch": f"B{label}",
                        "status": "complete"
                        if complete
                        else "incomplete"
                        if acquired
                        else "not_started",
                        "reason": None
                        if complete
                        else "Expected frequencies/rounds not fully acquired",
                        "acquired_shots": acquired,
                        "analyzed_shots": int(self.analysis.analyzed_shots[d, b]),
                    }
                )
        return {
            "branches": rows,
            "interpretation": "Thermal nominal preparations; all shots and full tones retained. Mean I/Q are diagnostic proxies, not bare-level assignments or leakage populations. No heralding, peak fit, Hamiltonian fit or calibration writeback.",
        }


class SpectroscopyAdapter(
    BaseAdapter[
        SpectroscopyCfg, SpectroscopyRun, SpectroscopyAnalyzeResult, NoAnalyzeParams
    ]
):
    """Fixed-capture Form, full raw acquisition, diagnostic analysis and restore."""

    exp_cls = SpectroscopyExp
    experiment_tag: ClassVar[str | None] = SpectroscopyExp.persistence_tag()
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[SpectroscopyCfg]:
        """Resolve role/tone identities and editable pulses from one Open/Reset capture."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: SpectroscopyCfg, *, context: RunContext
    ) -> SpectroscopyRun:
        """Keep original cfg with raw; caller checks cancel_signal.raise_if_error()."""
        del req
        return RunRecord(cfg=cfg, result=SpectroscopyExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[SpectroscopyRun, NoAnalyzeParams], *, plots: Plots
    ) -> SpectroscopyAnalyzeResult:
        """Use only source cfg/raw; never read current parameters or propose peaks."""
        return SpectroscopyAnalyzeResult(
            SpectroscopyExp().analyze(req.run_result, None, plots=plots)
        )
