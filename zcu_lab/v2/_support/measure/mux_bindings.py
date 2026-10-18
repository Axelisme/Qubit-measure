"""Mux component binding policy over the fixed parameter capture."""

from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.entry import RoleResolutionError

from zcu_lab.components.mux import MuxGroupSchema


def validate_mux_bindings(parameters: ParameterContext | None) -> None:
    """Validate optional qubit/resonator links in a fixed Open/Reset capture.

    Missing capture, unbound mux or absent qubits leaves resonator bring-up
    unrestricted. With qubits, require existing components of the correct kind
    and qubit.resonator matching the corresponding resonator. Structural group
    validation belongs to MuxGroupSchema. Source faults and mismatches raise
    with the field, original index and actual value. No live reads occur.
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


def require_mux_qubits(
    parameters: ParameterContext | None, tone_count: int | None = None
) -> tuple[str, ...]:
    """Return full qubit names after validating the fixed mux binding.

    parameters must be a fixed Open/Reset or Run/Load capture with a bound mux
    group, complete qubits and matching resonators. tone_count, when supplied,
    is the full (not active) numeric readout size and must equal the binding
    size. Missing capture/group/qubits or mismatches raise ValueError; alias
    and source faults propagate. Never reads live files or guesses identity.
    """
    if parameters is None:
        raise ValueError("Mux qubit experiments require a parameter capture")
    name = parameters.resolve_path("$mux")
    group = parameters.document().components.get(name)
    if not isinstance(group, MuxGroupSchema):
        raise ValueError(f"$mux={name!r}: expected group/mux")
    if group.qubits is None:
        raise ValueError(f"$mux.qubits is missing in {name!r}")
    validate_mux_bindings(parameters)
    if tone_count is not None and len(group.resonators) != tone_count:
        raise ValueError(
            f"$mux.resonators length={len(group.resonators)}; full readout tone count={tone_count}"
        )
    return tuple(group.qubits)
