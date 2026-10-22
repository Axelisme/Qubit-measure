"""Mux group readout slots, qubit identities and shared recovery initial values."""

from math import isfinite, pi

from zcu_tools.cfg_model import Expr, MuxReadoutSpec, Slot, Template, TemplateSeed
from zcu_tools.gui.form_adapter import FixedElements, FormContext

from zcu_lab.components.mux import MuxGroupSchema
from zcu_lab.templates import TEMPLATES

from .ctx_helpers import t1_initial_time
from .mux_bindings import require_mux_group, require_mux_qubits

READOUT_ELEMENTS = FixedElements(("modules", "readout", "module", "tones"))


def captured_mux_names(ctx: FormContext) -> tuple[str, ...]:
    """Require complete qubit/resonator bindings from the Open/Reset capture."""
    return require_mux_qubits(ctx.parameters)


def mux_readout_initial(
    ctx: FormContext, *, wait_for_ringup: bool = False
) -> Slot | Template:
    """Choose the readout source from ctx's fixed Open/Reset group capture.

    With a readout slot and wait_for_ringup=False, return a library reference,
    preserving all stored tones and scalar settings. Without a slot, seed the
    blank template with each resonator's calibration frequency and name label,
    generator from $mux.wiring.gen_ch, tone ADCs from $mux.wiring.ro_ch in group
    order, and gain 0.1. Wiring and frequency use Expr seeds. Missing/invalid
    sources leave required input blank. The operator confirms wiring, gain and
    timing before Run; durations remain unfilled. Missing capture, mux alias or
    group/mux raises ValueError; other alias/source faults propagate.

    wait_for_ringup opts spectroscopy into an editable Open/Reset timing seed.
    Ignore absent/null member kappas; all unset leaves trig_offset equal to
    the source time-of-flight offset. Otherwise require each set kappa to be
    finite and positive in MHz, or raise ValueError (alias/source faults
    propagate). Add 5/(2*pi*min(set kappas)) us to the captured scalar trigger
    offset, or the blank template's zero offset. A minimum kappa of 1 MHz adds
    approximately 0.795775 us. With a captured readout, preserve ro_length and
    seed length=max(original length, trig_offset+ro_length) in us. Without one,
    keep both required durations unfilled.
    This spectroscopy exception uses a detached slot copy when available and
    replaces tones from the group on both paths. Open/Reset leaves editable
    timing; subsequent cfg output retains the operator's edits. No source is
    mutated. Other mux consumers keep their original timing.
    """
    group = require_mux_group(ctx.parameters)
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("Mux forms require a parameter capture")
    has_readout = ctx.has("$mux.module.readout")
    if not wait_for_ringup and has_readout:
        return Slot("$mux.module.readout")
    tones: list[dict[str, TemplateSeed]] = []
    for index, resonator in enumerate(group.resonators):
        component = parameters.document().components.get(resonator)
        if component is None or component.kind.split("/")[0] != "resonator":
            raise ValueError(
                f"$mux.resonators[{index}]={resonator!r}: expected resonator"
            )
        tone: dict[str, TemplateSeed] = {
            "freq": Expr(f"{resonator}.freq"),
            "label": resonator,
        }
        if has_readout:
            # Preserve the existing ringup-copy defaults; only blank input changes.
            tone.update({"gain": 0.1, "ro_ch": index})
        else:
            tone.update({"gain": 0.1, "ro_ch": Expr(f"$mux.wiring.ro_ch[{index}]")})
        tones.append(tone)
    seeds: dict[str, TemplateSeed] = {"tones": tones}
    if wait_for_ringup:
        seeds.update(_ringup_timing(ctx, group))
    if has_readout:
        return Slot("$mux.module.readout", copy=True, set=seeds)
    seeds["gen_ch"] = Expr("$mux.wiring.gen_ch")
    return Template("readout/mux:blank", set=seeds)


def _ringup_timing(ctx: FormContext, group: MuxGroupSchema) -> dict[str, TemplateSeed]:
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("Mux timing requires a parameter capture")
    kappas: list[float] = []
    for index, resonator in enumerate(group.resonators):
        source = parameters.lookup_point(f"{resonator}.kappa")
        kappa = source.value
        if not source.present or kappa is None:
            continue
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
    wait_us = 5 / (2 * pi * min(kappas)) if kappas else 0.0
    if ctx.has("$mux.module.readout"):
        readout = parameters.library.module_spec(
            parameters.library_key("$mux.module.readout")
        )
        if not isinstance(readout, MuxReadoutSpec):
            raise TypeError("Mux readout slot must reference a MuxReadoutSpec")
        trigger = readout.trig_offset + wait_us
        length = max(readout.length, trigger + readout.ro_length)
    else:
        # Ringup can seed the trigger without inventing the required durations.
        offset = TEMPLATES["readout/mux:blank"].seeds["trig_offset"]
        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise TypeError("Blank mux trig_offset seed must be a numeric scalar")
        return {"trig_offset": offset + wait_us}
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
