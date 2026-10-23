"""Measure attachment for thermal four-state joint readout."""

from collections.abc import Sequence
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
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core import (
    BRANCH_LABELS,
    JointReadoutAnalysis,
    JointReadoutCfg,
    JointReadoutExp,
    JointReadoutResult,
)
from .form import make_form

JointReadoutRun: TypeAlias = RunRecord[JointReadoutCfg, JointReadoutResult]


@dataclass(frozen=True)
class JointReadoutAnalyzeResult:
    """JSON-safe summary of the saved-cfg-only joint classification."""

    analysis: JointReadoutAnalysis

    def to_summary_dict(self) -> dict[str, object]:
        """Distinguish complete, incomplete and unstarted branches without NaN JSON.

        Unanalysed counts/probabilities are None, never measured zeros. Counts
        include every shot of a complete branch; no diagonal-value gate applies.
        """
        rows: list[dict[str, object]] = []
        answer = self.analysis
        for index, label in enumerate(BRANCH_LABELS):
            complete = bool(answer.complete[index])
            acquired = int(answer.acquired_shots[index])
            rows.append(
                {
                    "branch": f"B{label}",
                    "status": "complete"
                    if complete
                    else "incomplete"
                    if acquired
                    else "not_started",
                    "reason": None
                    if complete
                    else "Expected rounds not fully acquired"
                    if acquired
                    else "No completed rounds acquired",
                    "acquired_shots": acquired,
                    "analyzed_shots": int(answer.analyzed_shots[index]),
                    "counts": answer.counts[index].astype(int).tolist()
                    if complete
                    else None,
                    "probabilities": answer.matrix[index].tolist()
                    if complete
                    else None,
                }
            )
        return {
            "branches": rows,
            "interpretation": "Nominal preparation → observed outcome, includes thermal population, preparation and readout effects. No heralding or inversion.",
        }


class JointReadoutAdapter(
    BaseAdapter[
        JointReadoutCfg, JointReadoutRun, JointReadoutAnalyzeResult, NoAnalyzeParams
    ]
):
    """Open, acquire, load and classify joint raw shots without calibration writes."""

    exp_cls = JointReadoutExp
    experiment_tag: ClassVar[str | None] = JointReadoutExp.persistence_tag()
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[JointReadoutCfg]:
        """Use one fixed capture for roles, full tones, pulses and GE centers."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: JointReadoutCfg, *, context: RunContext
    ) -> JointReadoutRun:
        """Pair the unchanged full cfg with raw IQ, including cancelled partial data.

        Context owns hardware, plots and the shared stop signal. The caller must
        check raise_if_error to distinguish acquisition errors from cancellation.
        """
        del req
        return RunRecord(cfg=cfg, result=JointReadoutExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JointReadoutRun, NoAnalyzeParams], *, plots: Plots
    ) -> JointReadoutAnalyzeResult:
        """Classify complete branches using saved cfg; never current parameters."""
        return JointReadoutAnalyzeResult(
            JointReadoutExp().analyze(req.run_result, None, plots=plots)
        )

    def get_writeback_items(
        self, req: WritebackRequest[JointReadoutRun, JointReadoutAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Return no proposals: this thermal matrix is a diagnostic, not a correction."""
        del req
        return []
