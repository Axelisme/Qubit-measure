"""Mux freq source/analysis/proposal hooks; numeric core remains GUI-free."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    RunRequest,
    SampleWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2._support.measure.mux_results import captured_qubits, indexed_summary
from zcu_lab.v2._support.mux import MuxFitFailure, MuxSweepResult
from zcu_lab.v2.twotone.freq.core import FreqAnalyzeOptions
from zcu_lab.v2.twotone.freq.gui import FreqAnalyzeParams

from .analysis import MuxFreqFit
from .core import MuxFreqCfg, MuxFreqExp
from .form import make_form

MuxFreqRunResult: TypeAlias = RunRecord[MuxFreqCfg, MuxSweepResult]


@dataclass(frozen=True)
class MuxFreqAnalyzeResult(AnalyzeResultBase):
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxFreqFit | MuxFitFailure, ...]
    names: tuple[str, ...] | None

    def to_summary_dict(self) -> dict[str, object]:
        """Present original positions/names, numerical values, quality or failure."""
        summaries: list[dict[str, object]] = []
        for outcome in self.outcomes:
            if isinstance(outcome, MuxFitFailure):
                values: dict[str, object] = {"failure": outcome.reason}
            else:
                values = {
                    "freq": outcome.freq,
                    "freq_err": outcome.freq_err,
                    "fwhm": outcome.fwhm,
                    "fwhm_err": outcome.fwhm_err,
                    "fit_quality": fit_quality_summary(
                        {"fit": outcome.quality},
                        summary_path=f"summary.tones[{len(summaries)}].fit_quality",
                    ),
                }
            summaries.append(indexed_summary(self.names, outcome.index, values))
        return {"tones": summaries}


class MuxFreqAdapter(
    BaseAdapter[MuxFreqCfg, MuxFreqRunResult, MuxFreqAnalyzeResult, FreqAnalyzeParams]
):
    """Run explicit sources and propose captured destinations without applying them."""

    experiment_tag: ClassVar[str | None] = MuxFreqExp.persistence_tag()
    exp_cls = MuxFreqExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MuxFreqCfg]:
        """Build the native list Form using the fixed Open/Reset capture."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: MuxFreqCfg, *, context: RunContext
    ) -> MuxFreqRunResult:
        """Pair the validated cfg with the core result using this operation's context."""
        del req
        return RunRecord(cfg, MuxFreqExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[MuxFreqRunResult, FreqAnalyzeParams], *, plots: Plots
    ) -> MuxFreqAnalyzeResult:
        """Analyze the explicit Run/Load source, retaining each original-index outcome."""
        params = req.analyze_params
        outcomes = MuxFreqExp().analyze(
            req.run_result,
            FreqAnalyzeOptions(model_type=params.model_type, plot_fit=params.plot_fit),
            plots=plots,
        )
        snapshot = req.run_result.cfg
        return MuxFreqAnalyzeResult(
            outcomes,
            None if snapshot is None else tuple(t.name for t in snapshot.tones),
        )

    def get_writeback_items(
        self, req: WritebackRequest[MuxFreqRunResult, MuxFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose only complete, successful tones using the same Run/Load capture.

        Missing capture returns no items. Captured bindings must match any source
        cfg names/count; errors propagate. Partial/nonfinite rows and failed fits
        produce no calibration items. Pulse proposals copy source cfg only, never
        current library values. This method neither writes nor applies proposals.
        """
        snapshot = req.run_result.cfg
        names = None if snapshot is None else tuple(t.name for t in snapshot.tones)
        qubits = captured_qubits(req.ctx.parameters, names)
        if qubits is None:
            return []
        items: list[WritebackItem] = []
        for row, outcome in enumerate(req.analyze_result.outcomes):
            if (
                isinstance(outcome, MuxFitFailure)
                or not np.isfinite(req.run_result.result.signals[row]).all()
            ):
                continue
            qubit = qubits[outcome.index]
            items.extend(
                [
                    ParameterWriteback(
                        target_name=f"{qubit}.freq",
                        description="Qubit frequency (MHz)",
                        proposed_value=outcome.freq,
                    ),
                    ParameterWriteback(
                        target_name=f"{qubit}.kappa",
                        description="Qubit linewidth FWHM (MHz)",
                        proposed_value=outcome.fwhm,
                    ),
                    SampleWriteback(
                        target_name=qubit,
                        description="Qubit frequency sample at captured bias",
                        proposed_value={
                            "freq_mhz": outcome.freq,
                            "freq_err_mhz": outcome.freq_err,
                        },
                    ),
                ]
            )
        return items
