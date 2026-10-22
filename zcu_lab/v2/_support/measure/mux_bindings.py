"""Mux component binding policy over the fixed parameter capture."""

from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.entry import RoleResolutionError

from zcu_lab.components.mux import MuxGroupSchema


def validate_mux_bindings(parameters: ParameterContext | None) -> None:
    """Validate optional mux qubit/resonator links in a fixed parameter capture.

    parameters is the borrowed, read-only ParameterContext captured for
    Open/Reset, or None when no capture is available. Return None without
    gating resonator bring-up if parameters is None, $mux is absent/unbound,
    its target is missing or not a MuxGroupSchema, or group.qubits is None.
    Callers that require a mux group, qubits or roles such as F1/F2 must enforce
    those requirements themselves.

    With qubits, validate the full ordered lists, not an active-tone subset:
    both components must exist, have qubit/resonator kinds respectively, and
    qubits[i].resonator must equal resonators[i]. ValueError diagnostics retain
    the field, original index and actual name, kind or resonator value.
    MuxGroupSchema owns structural validation, including names, equal lengths
    and duplicate qubits; the capture must contain model-validated components.

    Source failures and alias errors other than absent/unbound mux propagate.
    An unavailable capture is not treated as None. Reads use only the fixed
    capture, never live files, devices, current aliases or library assets.
    The capture is neither refreshed nor mutated.
    """
    if parameters is None:
        return
    try:
        name = parameters.resolve_path("$mux")
    except RoleResolutionError as exc:
        if exc.reason in ("Unknown alias", "Alias is not bound", "Role is not bound"):
            return
        raise
    components = parameters.document().components
    group = components.get(name)
    if not isinstance(group, MuxGroupSchema) or group.qubits is None:
        return
    for index, (qubit, resonator) in enumerate(
        zip(group.qubits, group.resonators, strict=True)
    ):
        for field, target, expected in (
            ("qubits", qubit, "qubit"),
            ("resonators", resonator, "resonator"),
        ):
            component = components.get(target)
            if component is None:
                raise ValueError(
                    f"$mux.{field}[{index}]={target!r}: component is missing"
                )
            if component.kind.split("/")[0] != expected:
                raise ValueError(
                    f"$mux.{field}[{index}]={target!r}: kind={component.kind!r}; expected {expected!r}"
                )
        actual = parameters.lookup_point(f"{qubit}.resonator")
        if not actual.present or actual.value != resonator:
            raise ValueError(
                f"$mux.qubits[{index}]={qubit!r}: {qubit}.resonator="
                f"{actual.value!r} (present={actual.present}); "
                f"expected $mux.resonators[{index}]={resonator!r}"
            )


def require_mux_group(parameters: ParameterContext | None) -> MuxGroupSchema:
    """Return the group/mux bound to scalar alias mux in a fixed capture.

    Missing capture, absent/unbound alias or a non-group target raises
    ValueError with the required component kind and alias. Other alias and
    source faults propagate. The capture is borrowed read-only.
    """
    requirement = "Mux forms require a group/mux component bound to alias mux"
    if parameters is None:
        raise ValueError(requirement)
    try:
        name = parameters.resolve_path("$mux")
    except RoleResolutionError as exc:
        if exc.reason in ("Unknown alias", "Alias is not bound", "Role is not bound"):
            raise ValueError(requirement) from exc
        raise
    group = parameters.document().components.get(name)
    if not isinstance(group, MuxGroupSchema):
        raise ValueError(f"{requirement}; $mux={name!r} is not group/mux")
    return group


def require_mux_qubits(
    parameters: ParameterContext | None, tone_count: int | None = None
) -> tuple[str, ...]:
    """Return full qubit names after validating the fixed mux binding.

    parameters must be a fixed Open/Reset or Run/Load capture with a bound mux
    group, complete qubits and matching resonators. tone_count, when supplied,
    is the full (not active) numeric readout size and must equal the binding
    size. Missing capture/group/qubits or mismatches raise ValueError; other
    alias and source faults propagate. Never reads live files or guesses identity.
    """
    group = require_mux_group(parameters)
    if group.qubits is None:
        raise ValueError("$mux.qubits is missing")
    validate_mux_bindings(parameters)
    if tone_count is not None and len(group.resonators) != tone_count:
        raise ValueError(
            f"$mux.resonators length={len(group.resonators)}; full readout tone count={tone_count}"
        )
    return tuple(group.qubits)
