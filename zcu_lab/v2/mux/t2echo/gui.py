"""Mux t2echo source/analysis/proposal hooks; numeric core remains GUI-free."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from zcu_tools.core.element import ElementIdentity
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

from zcu_lab.v2._support.measure.mux_results import (
    MuxCoherenceSummary,
    captured_qubits,
    coherence_summary,
)
from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2._support.mux_coherence import MuxCoherenceResult
from zcu_lab.v2._support.mux_coherence_analysis import MuxCoherenceFit
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoAnalyzeOptions
from zcu_lab.v2.twotone.time_domain.t2echo.gui import (
    T2EchoAnalyzeParams,
    coherence_review_note,
)

from .core import MuxT2EchoCfg, MuxT2EchoExp
from .form import make_form

MuxT2EchoRunResult: TypeAlias = RunRecord[MuxT2EchoCfg, MuxCoherenceResult]


@dataclass(frozen=True)
class MuxT2EchoAnalyzeResult(AnalyzeResultBase):
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxCoherenceFit | MuxFitFailure, ...]
    names: tuple[str, ...] | None

    def summary(self) -> MuxCoherenceSummary:
        """Return native coherence outcomes with fixed original tone identities."""
        return coherence_summary(self.outcomes, self.names)

    def to_summary_dict(self) -> dict[str, object]:
        """Serialize the owner summary for GUI and recipe consumers."""
        return self.summary().model_dump(mode="json")


class MuxT2EchoAdapter(
    BaseAdapter[
        MuxT2EchoCfg, MuxT2EchoRunResult, MuxT2EchoAnalyzeResult, T2EchoAnalyzeParams
    ]
):
    """Run explicit sources and propose captured destinations without applying them."""

    sample_device_alias: ClassVar[str | None] = "$mux.flux_source"
    exp_cls = MuxT2EchoExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MuxT2EchoCfg]:
        """Build the native list Form using the fixed Open/Reset capture."""
        return make_form(ctx)

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
        current library values. T2 and its sample share the coherence review gate:
        r2 >=0.9, normalized RMS <=0.1 and error/T2 <=0.5. Poor or unavailable
        quality leaves candidates unselected with reasons; users may override.
        This method neither writes nor applies proposals.
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
            note = coherence_review_note(outcome.t2, outcome.t2_err, outcome.quality)
            row_items: list[WritebackItem] = []
            row_items.extend(
                [
                    ParameterWriteback(
                        target_name=f"{qubit}.t2e",
                        description="t2e coherence time (us)",
                        proposed_value=outcome.t2,
                        selected=note is None,
                        note=note,
                    ),
                    SampleWriteback(
                        target_name=qubit,
                        description="t2e sample at captured bias",
                        proposed_value={
                            "t2e_us": outcome.t2,
                            "t2e_err_us": outcome.t2_err,
                        },
                        selected=note is None,
                        note=note,
                    ),
                ]
            )
            for item in row_items:
                item.element = ElementIdentity(index=outcome.index, name=qubit)
            items.extend(row_items)
        return items
