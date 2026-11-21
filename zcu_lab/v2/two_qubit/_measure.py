"""Fixed capture binding for two-qubit Forms, never used by core."""

from dataclasses import dataclass

from zcu_tools.cfg_model import MuxReadoutSpec
from zcu_tools.gui.form_adapter import FormContext
from zcu_tools.resources.entry import RoleResolutionError

from zcu_lab.components.mux import MuxGroupSchema
from zcu_lab.v2.support.measure.mux_bindings import validate_mux_bindings


@dataclass(frozen=True)
class CapturedQubits:
    """Component names and zero-based positions in the full mux readout.

    qubit1, qubit2 and mux are the captured component names. coupler is the
    captured component name only when the caller requires it, otherwise None.
    qubit1_tone and qubit2_tone follow $mux.qubits, not ADC numbers or labels.
    """

    qubit1: str
    qubit2: str
    mux: str
    coupler: str | None
    qubit1_tone: int
    qubit2_tone: int


def captured_qubits(
    ctx: FormContext, *, require_coupler: bool = False
) -> CapturedQubits:
    """Resolve positions and component names from a fixed Open/Reset capture.

    Require scalar $qubit1, $qubit2 and $mux aliases; only resolve $coupler
    when require_coupler is true. Both distinct qubits must appear in the
    ordered $mux.qubits list. If $mux.module.readout is bound, require a mux
    readout with one tone per listed qubit. Missing or inconsistent capture
    inputs raise ValueError with the relevant aliases or counts. Component
    kinds are not checked. The capture is neither refreshed nor changed.
    """
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("Bind $qubit1, $qubit2 and $mux in a fixed parameter capture")
    qubit1 = parameters.resolve_path("$qubit1")
    qubit2 = parameters.resolve_path("$qubit2")
    mux = parameters.resolve_path("$mux")
    coupler = parameters.resolve_path("$coupler") if require_coupler else None

    if qubit1 == qubit2:
        raise ValueError(
            f"$qubit1 and $qubit2 both bind {qubit1!r}; use distinct components"
        )
    qubits = parameters.lookup_point("$mux.qubits").value
    if not isinstance(qubits, list):
        raise ValueError(f"$mux={mux!r} must provide an ordered $mux.qubits list")
    for alias, name in (("$qubit1", qubit1), ("$qubit2", qubit2)):
        if name not in qubits:
            raise ValueError(f"$mux.qubits does not contain {alias}={name!r}")

    slot = "$mux.module.readout"
    if ctx.has(slot):
        readout = parameters.library.module_spec(parameters.library_key(slot))
        if not isinstance(readout, MuxReadoutSpec):
            raise ValueError(f"{slot} must provide a mux readout")
        if len(readout.tones) != len(qubits):
            raise ValueError(
                f"{slot} has {len(readout.tones)} tones; $mux.qubits has {len(qubits)} names"
            )

    return CapturedQubits(
        qubit1=qubit1,
        qubit2=qubit2,
        mux=mux,
        coupler=coupler,
        qubit1_tone=qubits.index(qubit1),
        qubit2_tone=qubits.index(qubit2),
    )


def captured_roles(ctx: FormContext) -> dict[str, str | int]:
    """Return readonly FTF names/tone positions from one borrowed Open/Reset capture.

    Require distinct fluxoniums, a transmon coupler and MuxGroup containing both.
    The shared mux validator owns qubit/resonator relations. Missing bindings or
    wrong kinds raise ValueError; source/alias failures propagate. No refresh,
    materialization, device access or mutation occurs.
    """
    validate_mux_bindings(ctx.parameters)
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("FTF readout requires a fixed parameter capture")
    f1, f2, coupler, mux = (
        parameters.resolve_path(f"${role}") for role in ("f1", "f2", "coupler", "mux")
    )
    for role, name, kind in (
        ("f1", f1, "qubit/fluxonium"),
        ("f2", f2, "qubit/fluxonium"),
        ("coupler", coupler, "qubit/transmon"),
    ):
        try:
            actual_kind = parameters.value(f"{name}.kind")
        except (AttributeError, RoleResolutionError) as exc:
            raise ValueError(f"${role}={name!r} must name a {kind} component") from exc
        if actual_kind != kind:
            raise ValueError(f"${role}={name!r} must name a {kind} component")
    if f1 == f2:
        raise ValueError("F1 and F2 must be distinct fluxonium components")
    group = parameters.component(mux)
    if not isinstance(group, MuxGroupSchema) or group.qubits is None:
        raise ValueError("FTF readout requires a MuxGroup with ordered qubits")
    if f1 not in group.qubits or f2 not in group.qubits:
        raise ValueError("Captured $mux.qubits must contain both F1 and F2")
    return {
        "f1": f1,
        "f2": f2,
        "coupler": coupler,
        "mux": mux,
        "f1_tone": group.qubits.index(f1),
        "f2_tone": group.qubits.index(f2),
        "tone_count": len(group.qubits),
    }
