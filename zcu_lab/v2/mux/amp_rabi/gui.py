"""Mux amp_rabi source/analysis/proposal hooks; numeric core remains GUI-free."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.core.element import ElementIdentity
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_models
from zcu_lab.v2._support.measure.mux_results import (
    MuxSummaryFailure,
    MuxSummaryTone,
    captured_qubits,
)
from zcu_lab.v2._support.measure.writeback_helpers import module_writeback
from zcu_lab.v2._support.mux import MuxFitFailure, MuxSweepResult
from zcu_lab.v2.twotone.rabi.amp_rabi.core import AmpRabiAnalyzeOptions
from zcu_lab.v2.twotone.rabi.amp_rabi.gui import AmpRabiAnalyzeParams

from .analysis import MuxAmpRabiFit
from .core import MuxAmpRabiCfg, MuxAmpRabiExp
from .form import make_form

MuxAmpRabiRunResult: TypeAlias = RunRecord[MuxAmpRabiCfg, MuxSweepResult]


class MuxAmpRabiFitSummary(MuxSummaryTone):
    """Native fit estimates in experiment units and qualified owner quality."""

    pi_gain: float | None = Field(
        description="Owner pi_gain in gain; null retains an unavailable estimate."
    )

    pi_gain_err: float | None = Field(
        description="Owner pi_gain_err in gain; null retains an unavailable estimate."
    )

    pi2_gain: float | None = Field(
        description="Owner pi2_gain in gain; null retains an unavailable estimate."
    )

    pi2_gain_err: float | None = Field(
        description="Owner pi2_gain_err in gain; null retains an unavailable estimate."
    )

    fit_quality: dict[str, FitQuality] | None


class MuxAmpRabiSummary(BaseModel):
    """Owner tone rows in analysis order, retaining successes and failures."""

    tones: list[MuxAmpRabiFitSummary | MuxSummaryFailure]


@dataclass(frozen=True)
class MuxAmpRabiAnalyzeResult(AnalyzeResultBase):
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxAmpRabiFit | MuxFitFailure, ...]
    names: tuple[str, ...] | None

    def summary(self) -> MuxAmpRabiSummary:
        """Return native tone estimates/failures with fixed identities and owner quality."""
        summaries: list[MuxAmpRabiFitSummary | MuxSummaryFailure] = []
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
                row = MuxAmpRabiFitSummary(
                    position=outcome.index,
                    name=name,
                    element=element,
                    pi_gain=outcome.pi_gain,
                    pi_gain_err=outcome.pi_gain_err,
                    pi2_gain=outcome.pi2_gain,
                    pi2_gain_err=outcome.pi2_gain_err,
                    fit_quality=fit_quality_models(
                        {"fit": outcome.quality},
                        summary_path=f"summary.tones[{len(summaries)}].fit_quality",
                    ),
                )
            summaries.append(row)
        return MuxAmpRabiSummary(tones=summaries)

    def to_summary_dict(self) -> dict[str, object]:
        """Serialize the same owner summary for GUI and recipe consumers."""
        return self.summary().model_dump(mode="json")


def _analyze(
    req: AnalyzeRequest[MuxAmpRabiRunResult, AmpRabiAnalyzeParams],
    *,
    plots: Plots,
) -> MuxAmpRabiAnalyzeResult:
    """Analyze the explicit Run/Load source, retaining each original-index outcome."""
    params = req.analyze_params
    outcomes = MuxAmpRabiExp().analyze(
        req.run_result, AmpRabiAnalyzeOptions(skip=params.skip), plots=plots
    )
    snapshot = req.run_result.cfg
    return MuxAmpRabiAnalyzeResult(
        outcomes,
        None if snapshot is None else tuple(t.name for t in snapshot.tones),
    )


def _get_writeback_items(
    req: WritebackRequest[MuxAmpRabiRunResult, MuxAmpRabiAnalyzeResult],
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
        if snapshot is None:
            continue
        pulse = snapshot.tones[outcome.index].require_drive()
        parameters = req.ctx.parameters
        if parameters is None:
            raise ValueError("Rabi pulse proposals require captured parameters")
        length = pulse.waveform.length
        if not isinstance(length, (int, float)):
            raise ValueError("Amplitude Rabi pi_product requires scalar source length")
        for suffix, slot, value, description in (
            ("pi_amp", "x180", outcome.pi_gain, "Pi pulse"),
            ("pi2_amp", "x90", outcome.pi2_gain, "Pi/2 pulse"),
        ):
            target = f"{qubit}.pulses.{suffix}"
            row_items.append(
                module_writeback(
                    pulse,
                    target=target,
                    description=description,
                    field_updates=(("gain", value),),
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
                proposed_value=outcome.pi_gain * length,
            )
        )
        for item in row_items:
            item.element = ElementIdentity(index=outcome.index, name=qubit)
        items.extend(row_items)
    return items


DEFINITION = Adapter(
    spec=MuxAmpRabiExp.AXES_SPEC,
    make_form=make_form,
    run=MuxAmpRabiExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: AmpRabiAnalyzeParams(),
        writeback=_get_writeback_items,
    ),
)
