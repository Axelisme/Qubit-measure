"""Mux t2echo source/analysis/proposal hooks; numeric core remains GUI-free."""

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
from zcu_lab.v2._support.mux_coherence import MuxCoherenceResult
from zcu_lab.v2._support.mux_coherence_analysis import MuxCoherenceFit
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoAnalyzeOptions
from zcu_lab.v2.twotone.time_domain.t2echo.gui import T2EchoAnalyzeParams

from .core import MuxT2EchoCfg, MuxT2EchoExp

MuxT2EchoRunResult: TypeAlias = RunRecord[MuxT2EchoCfg, MuxCoherenceResult]


@dataclass(frozen=True)
class MuxT2EchoAnalyzeResult(AnalyzeResultBase):
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxCoherenceFit | MuxFitFailure, ...]
    names: tuple[str, ...] | None

    def to_summary_dict(self) -> dict[str, object]:
        """Present original positions/names, numerical values, quality or failure."""
        summaries: list[dict[str, object]] = []
        for outcome in self.outcomes:
            if isinstance(outcome, MuxFitFailure):
                values: dict[str, object] = {"failure": outcome.reason}
            else:
                values = {
                    "t2": outcome.t2,
                    "t2_err": outcome.t2_err,
                    "detune": outcome.detune,
                    "detune_err": outcome.detune_err,
                    "fit_fringe": outcome.fit_fringe,
                    "fit_quality": fit_quality_summary({"fit": outcome.quality}),
                }
            summaries.append(indexed_summary(self.names, outcome.index, values))
        return {"tones": summaries}


class MuxT2EchoAdapter(
    BaseAdapter[
        MuxT2EchoCfg, MuxT2EchoRunResult, MuxT2EchoAnalyzeResult, T2EchoAnalyzeParams
    ]
):
    """Run explicit sources and propose captured destinations without applying them."""

    experiment_tag: ClassVar[str | None] = MuxT2EchoExp.persistence_tag()
    exp_cls = MuxT2EchoExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self, req: RunRequest, cfg: MuxT2EchoCfg, *, context: RunContext
    ) -> MuxT2EchoRunResult:
        """Pair the validated cfg with the core result using this operation's context."""
        del req
        return RunRecord(cfg, MuxT2EchoExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MuxT2EchoRunResult, T2EchoAnalyzeParams],
        *,
        plots: Plots,
    ) -> MuxT2EchoAnalyzeResult:
        """Analyze the explicit Run/Load source, retaining each original-index outcome."""
        params = req.analyze_params
        outcomes = MuxT2EchoExp().analyze(
            req.run_result,
            T2EchoAnalyzeOptions(
                fit_method=params.fit_method, fit_phase=params.fit_phase
            ),
            plots=plots,
        )
        snapshot = req.run_result.cfg
        return MuxT2EchoAnalyzeResult(
            outcomes,
            None if snapshot is None else tuple(t.name for t in snapshot.tones),
        )

    def get_writeback_items(
        self, req: WritebackRequest[MuxT2EchoRunResult, MuxT2EchoAnalyzeResult]
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
                        target_name=f"{qubit}.t2e",
                        description="t2e coherence time (us)",
                        proposed_value=outcome.t2,
                    ),
                    SampleWriteback(
                        target_name=qubit,
                        description="t2e sample at captured bias",
                        proposed_value={
                            "t2e_us": outcome.t2,
                            "t2e_err_us": outcome.t2_err,
                        },
                    ),
                ]
            )
        return items
