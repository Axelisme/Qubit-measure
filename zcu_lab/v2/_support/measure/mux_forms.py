"""Mux group readout slots, qubit identities and shared recovery initial values."""

from math import isfinite, pi

from zcu_tools.cfg_model import (
    Expr,
    FieldMeta,
    MuxReadoutSpec,
    Slot,
    Template,
    TemplateSeed,
)
from zcu_tools.gui.form_adapter import FixedElements, FormContext

from zcu_lab.components.mux import MuxGroupSchema

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
    order, and gain 0.1. Generator and frequency use Expr seeds; tone ADCs
    are detached integers from the captured group wiring because scalar Expr
    does not support indexing. Missing/invalid Expr sources leave required input
    blank. The operator confirms wiring, gain and timing before Run; durations
    remain unfilled unless wait_for_ringup is selected. Missing capture, mux alias or
    group/mux raises ValueError; other alias/source faults propagate.

    wait_for_ringup opts spectroscopy into an editable Open/Reset timing seed.
    Require each set kappa to be finite and positive in MHz, or raise ValueError
    (alias/source faults propagate). With a captured readout, ignore absent/null
    kappas, add 5/(2*pi*min(set kappas)) us (zero if all unset) to its trigger,
    preserve ro_length, and seed length=max(original length, trigger+ro_length).
    Without a slot, always seed ro_length=1.0 us, independent of kappa.
    Require all group kappas before seeding mixer_freq, trig_offset or length;
    otherwise leave those fields required and blank.
    Seed mixer_freq=(min(freq)+max(freq))/2 MHz when all tone frequencies exist.
    Seed trig_offset=max(time_of_fly)+5/(2*pi*min(kappas)) us and
    length=trig_offset+ro_length only when every resonator.ext.time_of_fly exists.
    Missing/null frequency or flight time leaves its dependent fields
    blank; invalid set values raise ValueError. These are editable candidates,
    not device-confirmed settings; capability checks remain at Run.
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
            tone.update({"gain": 0.1, "ro_ch": group.wiring.ro_ch[index]})
        tones.append(tone)
    seeds: dict[str, TemplateSeed] = {"tones": tones}
    if wait_for_ringup:
        seeds.update(_ringup_timing(ctx, group))
    if has_readout:
        return Slot("$mux.module.readout", copy=True, set=seeds)
    seeds["gen_ch"] = Expr("$mux.wiring.gen_ch")
    return Template("readout/mux:blank", set=seeds)


def mux_readout_seed_metadata(
    initial: Slot | Template,
) -> dict[tuple[str | int, ...], FieldMeta]:
    """Describe only the readout seeds actually authored by mux_readout_initial.

    initial is its returned Slot or Template, never a materialized cfg. Ref-only
    slots declare nothing. Generated gain 0.1 and timing overrides are uncalibrated
    candidates; untouched captured settings remain unknown. Paths are canonical
    for the modules/readout field. No sources, devices or calibration are read.
    """
    if isinstance(initial, Slot) and not initial.copy:
        return {}
    prefix = ("modules", "readout", "module")
    result: dict[tuple[str | int, ...], FieldMeta] = {}
    tones = initial.set.get("tones")
    if isinstance(tones, list):
        for index, tone in enumerate(tones):
            if tone.get("gain") == 0.1:
                result[(*prefix, "tones", index, "gain")] = FieldMeta(
                    description="Mux readout amplitude seed 0.1; operator confirmation required",
                    calibration="uncalibrated",
                )
    formulas = {
        "mixer_freq": "(min+max group resonator.freq)/2",
        "ro_length": "Fixed 1.0 us ADC window candidate; independent of group kappa",
        "trig_offset": "max group resonator.ext.time_of_fly + ringup",
        "length": "trig_offset + ro_length",
    }
    if isinstance(initial, Slot):
        formulas.update(
            {
                "trig_offset": "captured readout.trig_offset + 5/(2*pi*min set group kappa)",
                "length": "max(captured readout.length, trig_offset + captured readout.ro_length)",
            }
        )
    for field, formula in formulas.items():
        if field in initial.set:
            result[(*prefix, field)] = FieldMeta(
                description=formula, calibration="uncalibrated"
            )
    return result


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
    if not ctx.has("$mux.module.readout"):
        return _blank_ringup_timing(ctx, group, kappas)
    wait_us = 5 / (2 * pi * min(kappas)) if kappas else 0.0
    readout = parameters.library.module_spec(
        parameters.library_key("$mux.module.readout")
    )
    if not isinstance(readout, MuxReadoutSpec):
        raise TypeError("Mux readout slot must reference a MuxReadoutSpec")
    trigger = readout.trig_offset + wait_us
    length = max(readout.length, trigger + readout.ro_length)
    return {"trig_offset": trigger, "length": length}


def _blank_ringup_timing(
    ctx: FormContext, group: MuxGroupSchema, kappas: list[float]
) -> dict[str, TemplateSeed]:
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("Mux timing requires a parameter capture")
    seeds: dict[str, TemplateSeed] = dict.fromkeys(
        ("mixer_freq", "trig_offset", "ro_length", "length")
    )
    seeds["ro_length"] = 1.0
    source_values: dict[str, list[float]] = {}
    for field in ("freq", "ext.time_of_fly"):
        values: list[float] = []
        for resonator in group.resonators:
            source = parameters.lookup_point(f"{resonator}.{field}")
            value = source.value
            if not source.present or value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or (field == "ext.time_of_fly" and value < 0)
            ):
                requirement = (
                    "finite nonnegative us"
                    if field == "ext.time_of_fly"
                    else "finite MHz"
                )
                raise ValueError(f"{resonator}.{field}: expected {requirement}")
            values.append(value)
        source_values[field] = values

    if len(kappas) != len(group.resonators):
        return seeds
    wait_us = 5 / (2 * pi * min(kappas))
    frequencies = source_values["freq"]
    if len(frequencies) == len(group.resonators):
        seeds["mixer_freq"] = (min(frequencies) + max(frequencies)) / 2
    flight_times = source_values["ext.time_of_fly"]
    if len(flight_times) == len(group.resonators):
        trigger = max(flight_times) + wait_us
        seeds.update({"trig_offset": trigger, "length": trigger + 1.0})
    return seeds


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
