"""Captured identities and JSON consumers shared by mux qubit adapters."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.fitting.sampling import FringeSamplingQuality
from zcu_tools.gui.session.parameters import ParameterContext

from .mux_bindings import require_mux_qubits


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
    """Decode the original complete-list index without adding a summary identity."""

    index: int


class MuxSummaryFailure(MuxSummaryTone):
    """Native fitter failure reason at the original tone position."""

    reason: str


class MuxCoherenceFitSummary(MuxSummaryTone):
    """T2/error in us, detuning/error in MHz, fringe mode and owner diagnostics."""

    t2: float | None
    t2_err: float | None
    detune: float | None
    detune_err: float | None
    fit_fringe: bool
    sampling_quality: FringeSamplingQuality | None
    quality: FitQuality


class MuxCoherenceSummary(BaseModel):
    """Coherence outcomes in acquisition order, retaining failed original indices."""

    outcomes: list[MuxCoherenceFitSummary | MuxSummaryFailure]
    names: list[str] | None
