"""Mux t2ramsey source/analysis/proposal hooks; numeric core remains GUI-free."""

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
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyAnalyzeOptions
from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import T2RamseyAnalyzeParams

from .core import MuxT2RamseyCfg, MuxT2RamseyExp

MuxT2RamseyRunResult: TypeAlias = RunRecord[MuxT2RamseyCfg, MuxCoherenceResult]


@dataclass(frozen=True)
class MuxT2RamseyAnalyzeResult(AnalyzeResultBase):
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


class MuxT2RamseyAdapter(
    BaseAdapter[
        MuxT2RamseyCfg,
        MuxT2RamseyRunResult,
        MuxT2RamseyAnalyzeResult,
        T2RamseyAnalyzeParams,
    ]
):
    """Run explicit sources and propose captured destinations without applying them."""

    experiment_tag: ClassVar[str | None] = MuxT2RamseyExp.persistence_tag()
    exp_cls = MuxT2RamseyExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self, req: RunRequest, cfg: MuxT2RamseyCfg, *, context: RunContext
    ) -> MuxT2RamseyRunResult:
        """Pair the validated cfg with the core result using this operation's context."""
        del req
        return RunRecord(cfg, MuxT2RamseyExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MuxT2RamseyRunResult, T2RamseyAnalyzeParams],
        *,
        plots: Plots,
    ) -> MuxT2RamseyAnalyzeResult:
        """Analyze the explicit Run/Load source, retaining each original-index outcome."""
        params = req.analyze_params
        outcomes = MuxT2RamseyExp().analyze(
            req.run_result,
            T2RamseyAnalyzeOptions(
                fit_fringe=params.fit_fringe, fit_phase=params.fit_phase
            ),
            plots=plots,
        )
        snapshot = req.run_result.cfg
        return MuxT2RamseyAnalyzeResult(
            outcomes,
            None if snapshot is None else tuple(t.name for t in snapshot.tones),
        )

    def get_writeback_items(
        self, req: WritebackRequest[MuxT2RamseyRunResult, MuxT2RamseyAnalyzeResult]
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
                        target_name=f"{qubit}.t2r",
                        description="t2r coherence time (us)",
                        proposed_value=outcome.t2,
                    ),
                    SampleWriteback(
                        target_name=qubit,
                        description="t2r sample at captured bias",
                        proposed_value={
                            "t2r_us": outcome.t2,
                            "t2r_err_us": outcome.t2_err,
                        },
                    ),
                ]
            )
            data = req.run_result.result
            if (
                snapshot is not None
                and outcome.fit_fringe
                and np.isfinite(data.actual_detunes[row])
                and np.isfinite(data.source_frequencies[row])
                and np.isfinite(outcome.detune)
            ):
                frequency = (
                    data.source_frequencies[row]
                    + data.actual_detunes[row]
                    - outcome.detune
                )
                items.append(
                    ParameterWriteback(
                        target_name=f"{qubit}.freq",
                        description="Qubit frequency from sourced Ramsey fringe (MHz)",
                        proposed_value=float(frequency),
                    )
                )
        return items
