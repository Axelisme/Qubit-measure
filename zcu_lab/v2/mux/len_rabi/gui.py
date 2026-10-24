"""Mux len_rabi source/analysis/proposal hooks; numeric core remains GUI-free."""

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
from zcu_lab.v2._support.measure.writeback_helpers import module_writeback
from zcu_lab.v2._support.mux import MuxFitFailure, MuxSweepResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiAnalyzeOptions
from zcu_lab.v2.twotone.rabi.len_rabi.gui import LenRabiAnalyzeParams

from .analysis import MuxLenRabiFit
from .core import MuxLenRabiCfg, MuxLenRabiExp
from .form import make_form

MuxLenRabiRunResult: TypeAlias = RunRecord[MuxLenRabiCfg, MuxSweepResult]


class MuxLenRabiFitSummary(MuxSummaryTone):
    """Native fit estimates in experiment units and qualified owner quality."""

    pi_len: float | None = Field(
        description="Owner pi_len in us; null retains an unavailable estimate."
    )

    pi_len_err: float | None = Field(
        description="Owner pi_len_err in us; null retains an unavailable estimate."
    )

    pi2_len: float | None = Field(
        description="Owner pi2_len in us; null retains an unavailable estimate."
    )

    pi2_len_err: float | None = Field(
        description="Owner pi2_len_err in us; null retains an unavailable estimate."
    )

    rabi_f: float | None = Field(
        description="Owner rabi_f in MHz; null retains an unavailable estimate."
    )

    rabi_f_err: float | None = Field(
        description="Owner rabi_f_err in MHz; null retains an unavailable estimate."
    )

    fit_quality: dict[str, FitQuality] | None


class MuxLenRabiSummary(BaseModel):
    """Owner tone rows in analysis order, retaining successes and failures."""

    tones: list[MuxLenRabiFitSummary | MuxSummaryFailure]


@dataclass(frozen=True)
class MuxLenRabiAnalyzeResult(AnalyzeResultBase):
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxLenRabiFit | MuxFitFailure, ...]
    names: tuple[str, ...] | None

    def summary(self) -> MuxLenRabiSummary:
        """Return native tone estimates/failures with fixed identities and owner quality."""
        summaries: list[MuxLenRabiFitSummary | MuxSummaryFailure] = []
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
                row = MuxLenRabiFitSummary(
                    position=outcome.index,
                    name=name,
                    element=element,
                    pi_len=outcome.pi_len,
                    pi_len_err=outcome.pi_len_err,
                    pi2_len=outcome.pi2_len,
                    pi2_len_err=outcome.pi2_len_err,
                    rabi_f=outcome.rabi_f,
                    rabi_f_err=outcome.rabi_f_err,
                    fit_quality=fit_quality_models(
                        {"fit": outcome.quality},
                        summary_path=f"summary.tones[{len(summaries)}].fit_quality",
                    ),
                )
            summaries.append(row)
        return MuxLenRabiSummary(tones=summaries)

    def to_summary_dict(self) -> dict[str, object]:
        """Serialize the same owner summary for GUI and recipe consumers."""
        return self.summary().model_dump(mode="json")


class MuxLenRabiAdapter(
    BaseAdapter[
        MuxLenRabiCfg,
        MuxLenRabiRunResult,
        MuxLenRabiAnalyzeResult,
        LenRabiAnalyzeParams,
    ]
):
    """Run explicit sources and propose captured destinations without applying them."""

    exp_cls = MuxLenRabiExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MuxLenRabiCfg]:
        """Build the native list Form using the fixed Open/Reset capture."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: MuxLenRabiCfg, *, context: RunContext
    ) -> MuxLenRabiRunResult:
        """Pair the validated cfg with the core result using this operation's context."""
        del req
        return RunRecord(cfg, MuxLenRabiExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MuxLenRabiRunResult, LenRabiAnalyzeParams],
        *,
        plots: Plots,
    ) -> MuxLenRabiAnalyzeResult:
        """Analyze the explicit Run/Load source, retaining each original-index outcome."""
        params = req.analyze_params
        outcomes = MuxLenRabiExp().analyze(
            req.run_result,
            LenRabiAnalyzeOptions(decay=params.decay, fit_phase=params.fit_phase),
            plots=plots,
        )
        snapshot = req.run_result.cfg
        return MuxLenRabiAnalyzeResult(
            outcomes,
            None if snapshot is None else tuple(t.name for t in snapshot.tones),
        )

    def get_writeback_items(
        self, req: WritebackRequest[MuxLenRabiRunResult, MuxLenRabiAnalyzeResult]
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
            items.append(
                ParameterWriteback(
                    target_name=f"{qubit}.ext.rabi_freq",
                    description="Rabi oscillation frequency (MHz)",
                    proposed_value=outcome.rabi_f,
                    element=ElementIdentity(index=outcome.index, name=qubit),
                )
            )
            if snapshot is None:
                continue
            pulse = snapshot.tones[outcome.index].require_drive()
            parameters = req.ctx.parameters
            if parameters is None:
                raise ValueError("Rabi pulse proposals require captured parameters")
            gain = pulse.gain
            if not isinstance(gain, (int, float)):
                raise ValueError("Length Rabi pi_product requires scalar source gain")
            for suffix, slot, value, description in (
                ("pi_amp", "x180", outcome.pi_len, "Pi pulse"),
                ("pi2_amp", "x90", outcome.pi2_len, "Pi/2 pulse"),
            ):
                target = f"{qubit}.pulses.{suffix}"
                row_items.append(
                    module_writeback(
                        pulse,
                        target=target,
                        description=description,
                        field_updates=(("waveform.length", value),),
                    )
                )
                row_items.append(
                    ParameterWriteback(
                        target_name=f"{qubit}.module.{slot}",
                        description=f"{description} slot",
                        proposed_value=parameters.library_key(target),
                    )
                )
            row_items.append(
                ParameterWriteback(
                    target_name=f"{qubit}.ext.pi_product",
                    description="Pi pulse gain times length (a.u. * us)",
                    proposed_value=gain * outcome.pi_len,
                )
            )
            for item in row_items:
                item.element = ElementIdentity(index=outcome.index, name=qubit)
            items.extend(row_items)
        return items
