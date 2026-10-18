"""Mux t1 source/analysis/proposal hooks; numeric core remains GUI-free."""

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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2._support.measure.mux_results import captured_qubits, indexed_summary
from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2.twotone.time_domain.t1.core import T1AnalyzeOptions
from zcu_lab.v2.twotone.time_domain.t1.gui import T1AnalyzeParams

from .analysis import MuxT1Fit
from .core import MuxT1Cfg, MuxT1Exp, MuxT1Result

MuxT1RunResult: TypeAlias = RunRecord[MuxT1Cfg, MuxT1Result]


@dataclass(frozen=True)
class MuxT1AnalyzeResult(AnalyzeResultBase):
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxT1Fit | MuxFitFailure, ...]
    names: tuple[str, ...] | None

    def to_summary_dict(self) -> dict[str, object]:
        """Present original positions/names, numerical values, quality or failure."""
        summaries: list[dict[str, object]] = []
        for outcome in self.outcomes:
            if isinstance(outcome, MuxFitFailure):
                values: dict[str, object] = {"failure": outcome.reason}
            else:
                values = {
                    "t1": outcome.fit.t1,
                    "t1_err": outcome.fit.t1_err,
                    "t1b": outcome.fit.t1b,
                    "t1b_err": outcome.fit.t1b_err,
                    "fit_quality": fit_quality_summary(
                        outcome.fit.fit_quality,
                        summary_path=f"summary.tones[{len(summaries)}].fit_quality",
                    ),
                }
            summaries.append(indexed_summary(self.names, outcome.index, values))
        return {"tones": summaries}


class MuxT1Adapter(
    BaseAdapter[MuxT1Cfg, MuxT1RunResult, MuxT1AnalyzeResult, T1AnalyzeParams]
):
    """Run explicit sources and propose captured destinations without applying them."""

    experiment_tag: ClassVar[str | None] = MuxT1Exp.persistence_tag()
    exp_cls = MuxT1Exp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self, req: RunRequest, cfg: MuxT1Cfg, *, context: RunContext
    ) -> MuxT1RunResult:
        """Pair the validated cfg with the core result using this operation's context."""
        del req
        return RunRecord(cfg, MuxT1Exp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[MuxT1RunResult, T1AnalyzeParams], *, plots: Plots
    ) -> MuxT1AnalyzeResult:
        """Analyze the explicit Run/Load source, retaining each original-index outcome."""
        params = req.analyze_params
        outcomes = MuxT1Exp().analyze(
            req.run_result,
            T1AnalyzeOptions(skip=params.skip, dual_exp=params.dual_exp),
            plots=plots,
        )
        snapshot = req.run_result.cfg
        return MuxT1AnalyzeResult(
            outcomes,
            None if snapshot is None else tuple(t.name for t in snapshot.tones),
        )

    def get_writeback_items(
        self, req: WritebackRequest[MuxT1RunResult, MuxT1AnalyzeResult]
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
            fit = outcome.fit
            items.extend(
                [
                    ParameterWriteback(
                        target_name=f"{qubit}.t1",
                        description="T1 relaxation time (us)",
                        proposed_value=fit.t1,
                        stderr=fit.t1_err,
                    ),
                    SampleWriteback(
                        target_name=qubit,
                        description="T1 sample at captured bias",
                        proposed_value={"t1_us": fit.t1, "t1_err_us": fit.t1_err},
                    ),
                ]
            )
        return items
