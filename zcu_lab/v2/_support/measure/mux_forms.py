"""Mux group readout slots, qubit identities and shared recovery initial values."""

from collections.abc import Sequence
from math import isfinite, pi

from zcu_tools.cfg_model import (
    Expr,
    FieldMeta,
    FormNode,
    MuxReadoutSpec,
    Slot,
    Template,
    TemplateSeed,
)
from zcu_tools.gui.form_adapter import FixedElements, FormContext
from zcu_tools.gui.session.aliases import AliasTable
from zcu_tools.resources.entry import RoleResolutionError

from zcu_lab.components.mux import MuxGroupSchema

from .ctx_helpers import t1_initial_time
from .mux_bindings import require_mux_group, require_mux_qubits
from .seed_metadata import seed_metadata

READOUT_ELEMENTS = FixedElements(("readout", "module", "tones"))


def mux_tone_initials(
    tones: list[dict[str, FormNode]],
    participants: Sequence[int | str],
    *,
    fields: tuple[str, ...],
) -> list[FormNode]:
    """Build Open/Reset tone seeds after choosing original participant indices.

    tones is the complete ordered authored list. participants contains original
    indices; unknown captured names may remain as invalid draft strings and do
    not select a tone. Keep selected seeds and every identity field; set the
    listed nullable drive/sweep fields to None elsewhere.
    Return new dictionaries without editing the caller's trees. Ordinary edits
    never call this helper, so adding/removing participants preserves draft values.
    """
    return [
        {
            **tone,
            **({field: None for field in fields} if index not in participants else {}),
        }
        for index, tone in enumerate(tones)
    ]


def captured_mux_participants(
    ctx: FormContext, names: Sequence[str]
) -> list[int | str]:
    """Seed original indices from this Open/Reset capture's $participants.

    names is the complete captured qubit order for drive/flux Forms, or
    resonator order for onetone. Resolve scalar/list aliases and $ links through
    the alias owner. Missing or unbound rows (including []) select every index.
    Other alias/source faults propagate, including unbound nonempty elements.
    Preserve unknown names as ordinary invalid drafts so the field shows the
    name and rejects cfg until corrected. Never refresh sources or edit tones.
    """
    if ctx.parameters is None:
        raise ValueError("Mux participants require a parameter capture")
    try:
        value = AliasTable(ctx.parameters.aliases).resolve_value(
            "participants", ctx.parameters.document()
        )
    except RoleResolutionError as error:
        if error.reason not in ("Unknown alias", "Alias is not bound"):
            raise
        return list(range(len(names)))
    selected = [value] if isinstance(value, str) else value
    return [names.index(name) if name in names else name for name in selected]


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
    order, mixer from $mux.mixer_freq, and gain 0.1. Generator, mixer and
    frequency use Expr seeds; tone ADCs
    are detached integers from the captured group wiring because scalar Expr
    does not support indexing. When group ro_ch is None (not yet wired), tone
    ADCs stay blank and required, with missing seed status. Missing/invalid Expr
    sources also leave required input blank. The operator confirms wiring, gain
    and timing before Run. Durations remain unfilled unless wait_for_ringup is selected. Missing capture, mux alias or
    group/mux raises ValueError; other alias/source faults propagate.

    wait_for_ringup opts spectroscopy into an editable Open/Reset timing seed.
    Require each set kappa to be finite and positive in MHz, or raise ValueError
    (alias/source faults propagate). With a captured readout, ignore absent/null
    kappas, add 5/(2*pi*min(set kappas)) us (zero if all unset) to its trigger,
    preserve ro_length, and seed length=max(original length, trigger+ro_length).
    Without a slot, always seed ro_length=1.0 us, independent of kappa.
    Mixer uses $mux.mixer_freq independently of kappa and tone frequencies;
    missing/null mixer leaves that required field blank.
    Seed trig_offset=$mux.time_of_fly+5/(2*pi*min(kappas)) us and
    length=trig_offset+ro_length only when all group kappas and the group's
    flight time exist. Missing/null kappa or group flight time leaves trigger
    and length required and blank. Set flight time must be finite nonnegative
    us or raise ValueError. These are editable candidates,
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
            tone["gain"] = 0.1
            if group.wiring.ro_ch is not None:
                tone["ro_ch"] = group.wiring.ro_ch[index]
        tones.append(tone)
    seeds: dict[str, TemplateSeed] = {"tones": tones}
    if wait_for_ringup:
        seeds.update(_ringup_timing(ctx, group))
    if has_readout:
        return Slot("$mux.module.readout", copy=True, set=seeds)
    seeds["gen_ch"] = Expr("$mux.wiring.gen_ch")
    seeds["mixer_freq"] = Expr("$mux.mixer_freq")
    return Template("readout/mux:blank", set=seeds)


def mux_readout_seed_metadata(
    initial: Slot | Template,
) -> dict[tuple[str | int, ...], FieldMeta]:
    """Describe only the readout seeds actually authored by mux_readout_initial.

    initial is its returned Slot or Template, never a materialized cfg. Ref-only
    slots declare nothing. Only fixed gain 0.1 and ADC window 1.0 us are marked
    uncalibrated. Captured mixer and derived timing describe their source paths
    without claiming measured calibration. Paths are canonical
    for the modules/readout field. No sources, devices or calibration are read.
    """
    if isinstance(initial, Slot) and not initial.copy:
        return {}
    prefix = ("readout", "module")
    result: dict[tuple[str | int, ...], FieldMeta] = {}
    tones = initial.set.get("tones")
    if isinstance(tones, list):
        for index, tone in enumerate(tones):
            if tone.get("gain") == 0.1:
                result[(*prefix, "tones", index, "gain")] = seed_metadata(
                    "Mux readout amplitude seed 0.1 normalized; operator confirmation required",
                    fallback=True,
                )
    formulas = {
        "mixer_freq": "From captured $mux.mixer_freq in MHz",
        "ro_length": "Fixed 1.0 us ADC window candidate; independent of group kappa",
        "trig_offset": "From captured $mux.time_of_fly + 5/(2*pi*min($mux.resonators kappa)) in us",
        "length": "From captured $mux.time_of_fly + 5/(2*pi*min($mux.resonators kappa)) + fixed ro_length 1.0 us",
    }
    if isinstance(initial, Slot):
        formulas.update(
            {
                "trig_offset": "From captured $mux.module.readout.trig_offset + 5/(2*pi*min(set $mux.resonators kappa)) in us; no set kappa adds zero wait",
                "length": "From max(captured $mux.module.readout.length, derived trig_offset + captured $mux.module.readout.ro_length) in us",
            }
        )
    for field, formula in formulas.items():
        if field in initial.set:
            result[(*prefix, field)] = seed_metadata(
                formula, fallback=field == "ro_length"
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
        return _blank_ringup_timing(group, kappas)
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
    group: MuxGroupSchema, kappas: list[float]
) -> dict[str, TemplateSeed]:
    seeds: dict[str, TemplateSeed] = dict.fromkeys(
        ("trig_offset", "ro_length", "length")
    )
    seeds["ro_length"] = 1.0
    flight_time = group.time_of_fly
    if flight_time is None or len(kappas) != len(group.resonators):
        return seeds
    wait_us = 5 / (2 * pi * min(kappas))
    trigger = flight_time + wait_us
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
