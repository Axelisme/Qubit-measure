"""Mux len_rabi source/analysis/proposal hooks; numeric core remains GUI-free."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.core.element import ElementIdentity
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.mux_results import (
    MuxSummaryFailure,
    MuxSummaryTone,
    captured_qubits,
)
from zcu_lab.v2.support.measure.writeback_helpers import module_writeback
from zcu_lab.v2.support.mux import MuxFitFailure, MuxSweepResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiAnalyzeOptions

from .analysis import MuxLenRabiFit
from .core import MuxLenRabiCfg, MuxLenRabiExp
from .form import make_form

MuxLenRabiRunResult: TypeAlias = RunRecord[MuxLenRabiCfg, MuxSweepResult]


class MuxLenRabiFitSummary(MuxSummaryTone):
    """Decode native per-tone JSON estimates and the fitting owner diagnostics."""

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

    quality: FitQuality


class MuxLenRabiSummary(BaseModel):
    """Decode ordered native outcomes and the complete captured name list."""

    outcomes: list[MuxLenRabiFitSummary | MuxSummaryFailure]
    names: list[str] | None


@dataclass(frozen=True)
class MuxLenRabiAnalyzeResult:
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxLenRabiFit | MuxFitFailure, ...]
    names: tuple[str, ...] | None


def _MuxLenRabi_options(
    result: MuxLenRabiRunResult, ctx: SessionEnv
) -> LenRabiAnalyzeOptions:
    """Seed the analysis controls without changing the core defaults."""
    del result, ctx
    return LenRabiAnalyzeOptions(decay=True, fit_phase=False)


def _analyze(
    req: AnalyzeRequest[MuxLenRabiRunResult, LenRabiAnalyzeOptions], *, plots: Plots
) -> MuxLenRabiAnalyzeResult:
    """Analyze the explicit Run/Load source, retaining each original-index outcome."""
    outcomes = MuxLenRabiExp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )
    snapshot = req.run_result.cfg
    return MuxLenRabiAnalyzeResult(
        outcomes,
        None if snapshot is None else tuple(t.name for t in snapshot.tones),
    )


def _get_writeback_items(
    req: WritebackRequest[MuxLenRabiRunResult, MuxLenRabiAnalyzeResult],
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


def run(
    cfg: MuxLenRabiCfg, *, context: RunContext
) -> RunRecord[MuxLenRabiCfg, MuxSweepResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=MuxLenRabiExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=MuxLenRabiExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze, options=_MuxLenRabi_options, writeback=_get_writeback_items
    ),
)
