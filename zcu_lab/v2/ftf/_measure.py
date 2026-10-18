"""Fixed capture binding for the joint and idle FTF Forms, never used by core."""

from zcu_tools.cfg_model import Locked
from zcu_tools.gui.form_adapter import FormContext

from zcu_lab.components.mux import MuxGroupSchema
from zcu_lab.v2._support.measure.mux_bindings import validate_mux_bindings


def captured_roles(ctx: FormContext) -> dict[str, Locked]:
    """Return locked FTF names/tone positions from one borrowed Open/Reset capture.

    Require distinct fluxoniums, a transmon coupler and MuxGroup containing both.
    The shared mux validator owns qubit/resonator relations. Missing bindings or
    wrong kinds raise ValueError; source/alias failures propagate. No refresh,
    materialization, device access or mutation occurs.
    """
    validate_mux_bindings(ctx.parameters)
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("FTF readout requires a fixed parameter capture")
    components = parameters.document().components
    f1, f2, coupler, mux = (
        parameters.resolve_path(f"${role}") for role in ("f1", "f2", "coupler", "mux")
    )
    for role, name, kind in (
        ("f1", f1, "qubit/fluxonium"),
        ("f2", f2, "qubit/fluxonium"),
        ("coupler", coupler, "qubit/transmon"),
    ):
        component = components.get(name)
        if component is None or component.kind != kind:
            raise ValueError(f"${role}={name!r} must name a {kind} component")
    if f1 == f2:
        raise ValueError("F1 and F2 must be distinct fluxonium components")
    group = components.get(mux)
    if not isinstance(group, MuxGroupSchema) or group.qubits is None:
        raise ValueError("FTF readout requires a MuxGroup with ordered qubits")
    if f1 not in group.qubits or f2 not in group.qubits:
        raise ValueError("Captured $mux.qubits must contain both F1 and F2")
    return {
        "f1": Locked(f1),
        "f2": Locked(f2),
        "coupler": Locked(coupler),
        "mux": Locked(mux),
        "f1_tone": Locked(group.qubits.index(f1)),
        "f2_tone": Locked(group.qubits.index(f2)),
        "tone_count": Locked(len(group.qubits)),
    }
