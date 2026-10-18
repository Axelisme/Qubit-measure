"""Captured identities and indexed summaries shared by mux qubit adapters."""

from collections.abc import Mapping, Sequence

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


def indexed_summary(
    names: Sequence[str] | None, index: int, values: Mapping[str, object]
) -> dict[str, object]:
    """Attach original position/name to one outcome's presentation fields.

    names comes from the source cfg only. Without cfg the name is None, never
    guessed from current state. Invalid original positions raise IndexError.
    values contains the leaf's numeric scalars, fit_quality or failure reason;
    this helper neither computes fits nor changes proposal eligibility.
    """
    return {
        "position": index,
        "name": None if names is None else names[index],
        **values,
    }
