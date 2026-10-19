"""Captured identities and indexed summaries shared by mux qubit adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from zcu_tools.core.element import ElementIdentity
from zcu_tools.core.numeric_scalar import normalize_numpy_scalar
from zcu_tools.gui.session.parameters import ParameterContext

from zcu_lab.v2._support.mux import MuxFitFailure

from .analyze_results import fit_quality_summary
from .mux_bindings import require_mux_qubits

if TYPE_CHECKING:
    from zcu_lab.v2._support.mux_coherence_analysis import MuxCoherenceFit


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


def indexed_summary(
    names: Sequence[str] | None, index: int, values: Mapping[str, object]
) -> dict[str, object]:
    """Attach original position/name and author identity to one source outcome.

    names comes from the source cfg only. Without cfg the name is None, never
    guessed from current state. Invalid original positions raise IndexError.
    values contains the leaf's numeric scalars, JSON-safe fit_quality or failure
    reason. Direct NumPy scalars become Python scalars for nested analysis JSON;
    values and proposal eligibility are unchanged. Containers are not repaired.
    """
    identity = (
        {}
        if names is None
        else {"element": ElementIdentity(index=index, name=names[index])}
    )
    return {
        **identity,
        "position": index,
        "name": None if names is None else names[index],
        **{key: normalize_numpy_scalar(value) for key, value in values.items()},
    }


def coherence_summary(
    outcomes: Sequence[MuxCoherenceFit | MuxFitFailure],
    names: Sequence[str] | None,
) -> dict[str, object]:
    """Return {"tones": rows} for Ramsey/Echo without changing analysis row order.

    outcomes retains analysis row order and original tone indices. names is the
    complete source cfg name list, or None without cfg; unavailable names remain
    None and omit ElementIdentity. Invalid original positions raise IndexError
    when names is supplied. Successful rows copy t2/error in us, detune/error in
    MHz, the fringe flag and core quality. Failed rows store the fitter reason
    under failure. Direct NumPy scalars become Python scalars. Quality paths use output
    row offsets under summary.tones[row].fit_quality, not original tone indices.
    No fitting, quality calculation, source lookup or input mutation occurs.
    """
    summaries: list[dict[str, object]] = []
    for outcome in outcomes:
        if isinstance(outcome, MuxFitFailure):
            values: dict[str, object] = {"failure": outcome.reason}
        else:
            values = {
                "t2": outcome.t2,
                "t2_err": outcome.t2_err,
                "detune": outcome.detune,
                "detune_err": outcome.detune_err,
                "fit_fringe": outcome.fit_fringe,
                "fit_quality": fit_quality_summary(
                    {"fit": outcome.quality},
                    summary_path=f"summary.tones[{len(summaries)}].fit_quality",
                ),
            }
        summaries.append(indexed_summary(names, outcome.index, values))
    return {"tones": summaries}
