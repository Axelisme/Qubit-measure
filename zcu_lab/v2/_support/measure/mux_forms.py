"""Mux group readout slots, qubit identities and shared recovery initial values."""

from math import isfinite, pi

from zcu_tools.cfg_model import Expr, MuxReadoutSpec, Slot, Template, TemplateSeed
from zcu_tools.gui.form_adapter import FixedElements, FormContext

from zcu_lab.components.mux import MuxGroupSchema
from zcu_lab.templates import TEMPLATES

from .ctx_helpers import t1_initial_time
from .mux_bindings import require_mux_qubits

READOUT_ELEMENTS = FixedElements(("modules", "readout", "module", "tones"))


def captured_mux_names(ctx: FormContext) -> tuple[str, ...]:
    """Require complete qubit/resonator bindings from the Open/Reset capture."""
    return require_mux_qubits(ctx.parameters)


def mux_readout_initial(
    ctx: FormContext, *, wait_for_ringup: bool = False
) -> Slot | Template:
    """Seed one editable readout slot per captured group resonator.

    Use each resonator calibration frequency, gain=0.1 and its group-order
    index as ro_ch. Labels start as resonator names and remain editable text.
    Missing/invalid frequency leaves required input blank. Preserve an existing
    readout module's scalar settings via a detached copy; otherwise use the blank
    template's scalar initial values. Both paths replace tones from the group.
    The operator verifies the initial gain and channels against the lab wiring.

    wait_for_ringup opts spectroscopy into an editable Open/Reset timing seed.
    Require every member's kappa to be finite and positive in MHz, otherwise
    raise ValueError (alias/source faults propagate). Add 5/(2*pi*min(kappa)) us
    to the captured scalar trigger offset, or the blank template's zero offset.
    A minimum kappa of 1 MHz adds approximately 0.795775 us. Preserve ro_length
    and seed length=max(original length, trig_offset+ro_length) in us.
    Open/Reset leaves detached editable timing; subsequent cfg output retains
    the operator's edits. Other mux consumers keep their original timing.
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
    seeds: dict[str, TemplateSeed] = {"tones": tones}
    if wait_for_ringup:
        seeds.update(_ringup_timing(ctx, group))
    if ctx.has("$mux.module.readout"):
        return Slot("$mux.module.readout", copy=True, set=seeds)
    return Template("readout/mux:blank", set=seeds)


def _ringup_timing(ctx: FormContext, group: MuxGroupSchema) -> dict[str, TemplateSeed]:
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("Mux timing requires a parameter capture")
    kappas: list[float] = []
    for index, resonator in enumerate(group.resonators):
        source = parameters.lookup_point(f"{resonator}.kappa")
        kappa = source.value
        if (
            isinstance(kappa, bool)
            or not isinstance(kappa, (int, float))
            or not isfinite(kappa)
            or kappa <= 0
        ):
            raise ValueError(
                f"$mux.resonators[{index}]={resonator!r}: "
                "kappa must be finite and positive in MHz"
            )
        kappas.append(kappa)
    wait_us = 5 / (2 * pi * min(kappas))
    if ctx.has("$mux.module.readout"):
        readout = parameters.library.module_spec(
            parameters.library_key("$mux.module.readout")
        )
        if not isinstance(readout, MuxReadoutSpec):
            raise TypeError("Mux readout slot must reference a MuxReadoutSpec")
        trigger = readout.trig_offset + wait_us
        length = max(readout.length, trigger + readout.ro_length)
    else:
        # The blank template's public scalar seeds supply its timing baseline.
        blank = TEMPLATES["readout/mux:blank"].seeds
        timing: dict[str, float] = {}
        for field in ("trig_offset", "length", "ro_length"):
            value = blank[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Blank mux {field} seed must be a numeric scalar")
            timing[field] = value
        trigger = timing["trig_offset"] + wait_us
        length = max(timing["length"], trigger + timing["ro_length"])
    return {"trig_offset": trigger, "length": length}


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
