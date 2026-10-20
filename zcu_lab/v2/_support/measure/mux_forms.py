"""Mux group readout slots, qubit identities and shared recovery initial values."""

from zcu_tools.cfg_model import Expr, Slot, Template, TemplateSeed
from zcu_tools.gui.form_adapter import FixedElements, FormContext

from zcu_lab.components.mux import MuxGroupSchema

from .ctx_helpers import t1_initial_time
from .mux_bindings import require_mux_qubits

READOUT_ELEMENTS = FixedElements(("modules", "readout", "module", "tones"))


def captured_mux_names(ctx: FormContext) -> tuple[str, ...]:
    """Require complete qubit/resonator bindings from the Open/Reset capture."""
    return require_mux_qubits(ctx.parameters)


def mux_readout_initial(ctx: FormContext) -> Slot | Template:
    """Seed one editable readout slot per captured group resonator.

    Use each resonator calibration frequency, gain=0.1 and its group-order
    index as ro_ch. Labels start as resonator names but remain ordinary editable
    text, not identities. Missing/invalid frequency leaves required input blank.
    Preserve an existing readout module's scalar settings via a detached copy;
    otherwise use the blank template's scalar initial values. Both paths replace
    tones from the group. No materialization, hardware query or writeback occurs.
    The initial gain and channels are not a hardware safety guarantee.
    """
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("Mux Forms require a parameter capture")
    name = parameters.resolve_path("$mux")
    group = parameters.document().components.get(name)
    if not isinstance(group, MuxGroupSchema):
        raise ValueError(f"$mux={name!r}: expected group/mux")
    tones: list[dict[str, TemplateSeed]] = []
    for index, resonator in enumerate(group.resonators):
        component = parameters.document().components.get(resonator)
        if component is None or component.kind.split("/")[0] != "resonator":
            raise ValueError(
                f"$mux.resonators[{index}]={resonator!r}: expected resonator"
            )
        tones.append(
            {
                "freq": Expr(f"{resonator}.freq"),
                "gain": 0.1,
                "ro_ch": index,
                "label": resonator,
            }
        )
    if ctx.has("$mux.module.readout"):
        return Slot("$mux.module.readout", copy=True, set={"tones": tones})
    return Template("readout/mux:blank", set={"tones": tones})


def mux_relax_delay(ctx: FormContext, names: tuple[str, ...]) -> float | None:
    """Return five times the largest captured T1, in us, for shared recovery.

    names is the complete validated capture-ordered qubit tuple. Each missing
    T1 uses the common 30 us initial scale; any stored invalid T1 leaves the
    required delay blank. Uses the same T1 policy as single-qubit Forms,
    without refreshing sources or writing calibration.
    """
    delays = [
        t1_initial_time(ctx.parameters, multiplier=5.0, qubit=name) for name in names
    ]
    if any(delay is None for delay in delays):
        return None
    return max(delay for delay in delays if delay is not None)
