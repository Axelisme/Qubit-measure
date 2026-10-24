"""Captured identities and indexed summaries shared by mux qubit adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.fitting.sampling import FringeSamplingQuality
from zcu_tools.core.element import ElementIdentity
from zcu_tools.gui.session.parameters import ParameterContext

from zcu_lab.v2.support.mux import MuxFitFailure

from .analyze_results import fit_quality_models
from .mux_bindings import require_mux_qubits

if TYPE_CHECKING:
    from zcu_lab.v2.support.mux_coherence_analysis import MuxCoherenceFit


def captured_qubits(
    parameters: ParameterContext | None, names: Sequence[str] | None
) -> tuple[str, ...] | None:
    """Resolve full qubit destinations from this Run/Load capture only.

    names is the complete ordered cfg tone identity list, or None when the
    source has no cfg. Missing capture returns None, permitting pure analysis
    but no proposals. With a capture, enforce full mux binding and matching cfg
    names/count; invalid bindings or source association raise ValueError. Never
    read a current Form, library or global alias to repair missing history.
    """
    if parameters is None:
        return None
    qubits = require_mux_qubits(parameters, None if names is None else len(names))
    if names is not None and tuple(names) != qubits:
        raise ValueError(
            f"source tones names={tuple(names)!r}; captured qubits={qubits!r}"
        )
    return qubits


class MuxSummaryTone(BaseModel):
    """Original complete-list position/name; element is absent without source names."""

    position: int
    name: str | None
    element: ElementIdentity | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class MuxSummaryFailure(MuxSummaryTone):
    """Native fitter failure reason at the original tone position."""

    failure: str


class MuxCoherenceFitSummary(MuxSummaryTone):
    """T2/error in us, detuning/error in MHz, fringe mode and owner diagnostics."""

    t2: float
    t2_err: float
    detune: float
    detune_err: float
    fit_fringe: bool
    sampling_quality: FringeSamplingQuality | None
    fit_quality: dict[str, FitQuality] | None


class MuxCoherenceSummary(BaseModel):
    """Coherence outcomes in acquisition order, retaining failed original indices."""

    tones: list[MuxCoherenceFitSummary | MuxSummaryFailure]


def coherence_summary(
    outcomes: Sequence[MuxCoherenceFit | MuxFitFailure],
    names: Sequence[str] | None,
) -> MuxCoherenceSummary:
    """Project native coherence outcomes and qualified quality paths without refitting.

    Names come from the full source cfg, never live state. Each original index
    retains its acquisition row; invalid positions raise IndexError with names.
    """
    summaries: list[MuxCoherenceFitSummary | MuxSummaryFailure] = []
    for outcome in outcomes:
        name = None if names is None else names[outcome.index]
        element = (
            None if name is None else ElementIdentity(index=outcome.index, name=name)
        )
        if isinstance(outcome, MuxFitFailure):
            row = MuxSummaryFailure(
                position=outcome.index,
                name=name,
                element=element,
                failure=outcome.reason,
            )
        else:
            row = MuxCoherenceFitSummary(
                position=outcome.index,
                name=name,
                element=element,
                t2=outcome.t2,
                t2_err=outcome.t2_err,
                detune=outcome.detune,
                detune_err=outcome.detune_err,
                fit_fringe=outcome.fit_fringe,
                sampling_quality=outcome.sampling_quality,
                fit_quality=fit_quality_models(
                    {"fit": outcome.quality},
                    summary_path=f"summary.tones[{len(summaries)}].fit_quality",
                ),
            )
        summaries.append(row)
    return MuxCoherenceSummary(tones=summaries)
