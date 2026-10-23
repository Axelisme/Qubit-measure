"""Mux t1 source/analysis/proposal hooks; numeric core remains GUI-free."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
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

from zcu_lab.v2._support.measure.analyze_results import fit_quality_models
from zcu_lab.v2._support.measure.mux_results import (
    MuxSummaryFailure,
    MuxSummaryTone,
    captured_qubits,
)
from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2.twotone.time_domain.t1.core import T1AnalyzeOptions
from zcu_lab.v2.twotone.time_domain.t1.gui import T1AnalyzeParams

from .analysis import MuxT1Fit
from .core import MuxT1Cfg, MuxT1Exp, MuxT1Result
from .form import make_form

MuxT1RunResult: TypeAlias = RunRecord[MuxT1Cfg, MuxT1Result]


class MuxT1FitSummary(MuxSummaryTone):
    """Native fit estimates in experiment units and qualified owner quality."""

    t1: float | None = Field(
        description="Owner t1 in us; null retains an unavailable estimate."
    )

    t1_err: float | None = Field(
        description="Owner t1_err in us; null retains an unavailable estimate."
    )

    t1b: float | None = Field(
        description="Owner t1b in us; null retains an unavailable estimate."
    )

    t1b_err: float | None = Field(
        description="Owner t1b_err in us; null retains an unavailable estimate."
    )

    fit_quality: dict[str, FitQuality] | None


class MuxT1Summary(BaseModel):
    """Owner tone rows in analysis order, retaining successes and failures."""

    tones: list[MuxT1FitSummary | MuxSummaryFailure]


@dataclass(frozen=True)
class MuxT1AnalyzeResult(AnalyzeResultBase):
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxT1Fit | MuxFitFailure, ...]
    names: tuple[str, ...] | None

    def summary(self) -> MuxT1Summary:
        """Return native tone estimates/failures with fixed identities and owner quality."""
        summaries: list[MuxT1FitSummary | MuxSummaryFailure] = []
        for outcome in self.outcomes:
            name = None if self.names is None else self.names[outcome.index]
            element = (
                None
                if name is None
                else ElementIdentity(index=outcome.index, name=name)
            )
            if isinstance(outcome, MuxFitFailure):
                row = MuxSummaryFailure(
                    position=outcome.index,
                    name=name,
                    element=element,
                    failure=outcome.reason,
                )
            else:
                row = MuxT1FitSummary(
                    position=outcome.index,
                    name=name,
                    element=element,
                    t1=outcome.fit.t1,
                    t1_err=outcome.fit.t1_err,
                    t1b=outcome.fit.t1b,
                    t1b_err=outcome.fit.t1b_err,
                    fit_quality=fit_quality_models(
                        outcome.fit.fit_quality,
                        summary_path=f"summary.tones[{len(summaries)}].fit_quality",
                    ),
                )
            summaries.append(row)
        return MuxT1Summary(tones=summaries)

    def to_summary_dict(self) -> dict[str, object]:
        """Serialize the same owner summary for GUI and recipe consumers."""
        return self.summary().model_dump(mode="json")


class MuxT1Adapter(
    BaseAdapter[MuxT1Cfg, MuxT1RunResult, MuxT1AnalyzeResult, T1AnalyzeParams]
):
    """Run explicit sources and propose captured destinations without applying them."""

    experiment_tag: ClassVar[str | None] = MuxT1Exp.persistence_tag()
    sample_device_alias: ClassVar[str | None] = "$mux.flux_source"
    exp_cls = MuxT1Exp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MuxT1Cfg]:
        """Build the native list Form using the fixed Open/Reset capture."""
        return make_form(ctx)

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
            row_items: list[WritebackItem] = []
            fit = outcome.fit
            row_items.extend(
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
            for item in row_items:
                item.element = ElementIdentity(index=outcome.index, name=qubit)
            items.extend(row_items)
        return items
