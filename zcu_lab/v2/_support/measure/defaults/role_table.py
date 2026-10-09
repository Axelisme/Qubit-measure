"""The role vocabulary as data + two generic builders.

Every role's default is a single :class:`RoleDef` literal — its module/waveform
shape, the per-sub-section pulse/ro seeds, and its library adopt policy — so
"what does role X default to" is one table cell instead of a per-role factory
file. Two generic builders consume the data:

- :func:`role_blank` assembles the blank value tree by composing the *verbatim*
  ``patch_pulse_fields`` / ``patch_ro_cfg_fields`` / ``make_trig_offset`` helpers
  over the L1 blank shape — replacing the per-role ``make_<role>_default`` bodies.
- :func:`role_ref` does the library-aware adopt-or-(blank|None) — one helper
  replacing the 8 near-identical ``make_<role>_ref_default`` clones (and their
  ``@overload`` stubs).

The seed value carriers (EvalValue / :class:`Source` / :data:`TRIG`) wrap
canonical parameter references and the read-only value-source escape hatch. They resolve
against the whole ``SessionEnv`` (predictor / device / project / …) while still
materializing to direct/eval scalar leaves before the cfg tree leaves the builder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES
from zcu_tools.gui.app.measure.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.cfg import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceValue,
    ScalarValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.gui.session.value_lookup import ValueRef, resolve_value_ref
from zcu_tools.program.v2.modules import AbsResetCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.pulse import PulseCfg

from zcu_lab.v2._support.measure.defaults.helpers import (
    make_trig_offset,
    patch_pulse_fields,
    patch_ro_cfg_fields,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.cfg import CfgSectionSpec

_RefNode = ReferenceValue


def make_pulse_spec_() -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)


def make_direct_readout_spec_() -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("readout/direct").make_spec(MAIN_PROGRAM_SPEC_POLICY)


def make_pulse_readout_spec_() -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("readout/pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)


def make_none_reset_spec_() -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("reset/none").make_spec(MAIN_PROGRAM_SPEC_POLICY)


def make_pulse_reset_spec_() -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("reset/pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)


def make_two_pulse_reset_spec_() -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("reset/two_pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)


def make_bath_reset_spec_() -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("reset/bath").make_spec(MAIN_PROGRAM_SPEC_POLICY)


def make_const_waveform_spec_() -> CfgSectionSpec:
    return PROGRAM_SHAPES.waveform("const").make_spec(MAIN_PROGRAM_SPEC_POLICY)


def make_cosine_waveform_spec_() -> CfgSectionSpec:
    return PROGRAM_SHAPES.waveform("cosine").make_spec(MAIN_PROGRAM_SPEC_POLICY)


# ---------------------------------------------------------------------------
# Deferred calibration and registered value-source seeds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """A read-only value-source seed resolved once while building a role default."""

    key: str
    type_name: str | None = None


class _Trig(Enum):
    TRIG = "trig"


#: Deferred trigger time of flight in the resonator calibration.
TRIG = _Trig.TRIG

#: A seed value in a RoleDef: a raw scalar, a deferred EvalValue, a registered
#: ``Source``, or the ``TRIG`` rule (ro_cfg only).
SeedVal = float | int | str | bool | EvalValue | Source | _Trig


def _resolve(ctx: SessionEnv, v: SeedVal) -> ScalarValue:
    """Lower a seed value to a scalar leaf (DirectValue / EvalValue)."""
    if isinstance(v, EvalValue):
        return EvalValue(v.expr)
    if isinstance(v, Source):
        return DirectValue(resolve_value_ref(ValueRef(v.key, v.type_name), ctx.values))
    if v is TRIG:
        return make_trig_offset(
            ctx, trig_expr="$resonator.ext.time_of_fly + 0.05", trig_fallback=0.55
        )
    return DirectValue(v)


# ---------------------------------------------------------------------------
# Role description — pure data.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lib:
    """A library adopt policy: a module type (``None`` → waveform library) and the
    canonical module slot paths in priority order."""

    module_type: type | None
    prefer: tuple[str, ...]


@dataclass(frozen=True)
class Pulse:
    """A pulse sub-section's seed (applied via ``patch_pulse_fields``).

    ``at`` is the value-tree path of the pulse section: ``""`` for a flat pulse
    (qub_probe), or a nested key (``"pulse_cfg"``, ``"pulse1_cfg"``,
    ``"cavity_tone_cfg"``, …) for a composite module.
    """

    freq: SeedVal
    ch: SeedVal
    gain: SeedVal
    length: SeedVal
    at: str = ""


@dataclass(frozen=True)
class Ref:
    """A nested module/waveform ref inside a composite role seeded from another role."""

    at: str
    role_id: str
    optional: bool = False


@dataclass(frozen=True)
class Ro:
    """A readout ``ro_cfg`` seed (applied via ``patch_ro_cfg_fields``); the
    trig-offset is always the ``TRIG`` rule. ``at`` is ``""`` for a bare
    DirectReadout or ``"ro_cfg"`` for a pulse-readout's sub-section."""

    ro_freq: SeedVal
    ro_ch: SeedVal
    ro_length: SeedVal = 0.9
    at: str = "ro_cfg"


@dataclass(frozen=True)
class RoleDef:
    """One role's default, as data. ``shape`` is the spec factory for the blank
    value tree; the seeds overwrite it; ``lib`` (when set) is the adopt policy."""

    shape: Callable[[], CfgSectionSpec]
    tag: str
    pulses: tuple[Pulse, ...] = ()
    refs: tuple[Ref, ...] = ()
    ro: Ro | None = None
    wf_length: SeedVal | None = None
    adopt_waveform: str | None = None
    is_waveform: bool = False
    lib: Lib | None = None


# ---------------------------------------------------------------------------
# Generic builders — replace the per-role factory bodies + the 8 ref clones.
# ---------------------------------------------------------------------------


def _section(value: CfgSectionValue, at: str) -> CfgSectionValue:
    if at == "":
        return value
    sub = value.fields.get(at)
    if isinstance(sub, ReferenceValue):
        return sub.value
    if not isinstance(sub, CfgSectionValue):
        raise RuntimeError(f"role seed path {at!r} is not a section")
    return sub


def _nested_ref(role_id: str, ctx: SessionEnv, *, optional: bool) -> _RefNode | None:
    try:
        role = ROLE_TABLE[role_id]
    except KeyError as exc:
        raise RuntimeError(f"unknown nested role {role_id!r}") from exc
    if role.lib is None:
        if optional:
            return None
        return role_blank(role, ctx)
    return role_ref(role, ctx, optional=optional)


def role_blank(role: RoleDef, ctx: SessionEnv) -> _RefNode:
    """Assemble a role's blank value tree from its data (never a library lookup)."""
    value = make_default_value(role.shape())

    for ref in role.refs:
        value.fields[ref.at] = _nested_ref(ref.role_id, ctx, optional=ref.optional)

    for p in role.pulses:
        patch_pulse_fields(
            _section(value, p.at),
            freq=_resolve(ctx, p.freq),
            ch=_resolve(ctx, p.ch),
            gain=_resolve(ctx, p.gain),
            length=_resolve(ctx, p.length),
        )

    if role.ro is not None:
        patch_ro_cfg_fields(
            _section(value, role.ro.at),
            ro_freq=_resolve(ctx, role.ro.ro_freq),
            ro_ch=_resolve(ctx, role.ro.ro_ch),
            trig_offset=_resolve(ctx, TRIG),
            ro_length=_resolve(ctx, role.ro.ro_length),
        )

    if role.wf_length is not None:
        value.with_field("length", _resolve(ctx, role.wf_length))

    if role.is_waveform:
        return ReferenceValue(chosen_key=role.tag, value=value)
    return ReferenceValue(role.tag, value)


def role_ref(
    role: RoleDef, ctx: SessionEnv, *, optional: bool = False
) -> _RefNode | None:
    """Mount a canonical slot, retaining SMART's optional/blank fallback.

    Missing aliases remain linked empty until refresh binds them. Bound unset
    slots use None for optional roles or a fresh inline role otherwise. Source,
    malformed module and materialization failures propagate. No legacy names
    or live session library are consulted.
    """
    from zcu_lab.v2._support.measure.seeds import custom, module_seed

    lib = role.lib
    if lib is None or lib.module_type is None:
        raise RuntimeError(f"role {role.tag!r} has no module slot policy")
    fallback = (
        None
        if optional
        else custom(lambda env: role_blank(role, env), description=f"inline:{role.tag}")
    )
    return module_seed(lib.prefer[0], fallback=fallback).resolve(ctx)


# ---------------------------------------------------------------------------
# ROLE_TABLE — the single declarative source of every role default.
# ---------------------------------------------------------------------------

# Shared inline qubit pulse reads canonical scalar and wiring calibration.
_QUB_CH = EvalValue("$qubit.wiring.L01.ch")
_QUB_PULSE: tuple[Pulse, ...] = (Pulse(EvalValue("$qubit.freq"), _QUB_CH, 0.1, 5.1),)

ROLE_TABLE: dict[str, RoleDef] = {
    # qubit / resonator probe pulses
    "qub_probe": RoleDef(
        make_pulse_spec_,
        make_custom_reference_key("Pulse"),
        _QUB_PULSE,
        lib=Lib(PulseCfg, ("$qubit.module.probe",)),
    ),
    "res_probe": RoleDef(
        make_pulse_spec_,
        make_custom_reference_key("Pulse"),
        (
            Pulse(
                EvalValue("$resonator.freq"),
                EvalValue("$resonator.wiring.ch"),
                0.05,
                1.0,
            ),
        ),
        lib=Lib(PulseCfg, ("$resonator.module.probe",)),
    ),
    "pi_pulse": RoleDef(
        make_pulse_spec_,
        make_custom_reference_key("Pulse"),
        _QUB_PULSE,
        lib=Lib(PulseCfg, ("$qubit.module.x180",)),
    ),
    "rabi_pulse": RoleDef(
        make_pulse_spec_,
        make_custom_reference_key("Pulse"),
        _QUB_PULSE,
        lib=Lib(PulseCfg, ("$qubit.module.x180",)),
    ),
    "pi2_pulse": RoleDef(
        make_pulse_spec_,
        make_custom_reference_key("Pulse"),
        _QUB_PULSE,
        lib=Lib(PulseCfg, ("$qubit.module.x90",)),
    ),
    # readout: "readout" is the library-aware pulse readout (ModuleInit.SMART prefers a
    # calibrated library entry; ModuleInit.INLINE seeds a fresh inline pulse readout —
    # the role formerly split out as "pulse_readout"). direct_readout / readout_dpm
    # are the other inline-only shapes.
    "readout": RoleDef(
        make_pulse_readout_spec_,
        make_custom_reference_key("Pulse Readout"),
        (
            Pulse(
                EvalValue("$resonator.freq"),
                EvalValue("$resonator.wiring.ch"),
                0.1,
                1.0,
                at="pulse_cfg",
            ),
        ),
        ro=Ro(EvalValue("$resonator.freq"), EvalValue("$resonator.wiring.ro_ch")),
        lib=Lib(PulseReadoutCfg, ("$qubit.module.readout",)),
    ),
    "direct_readout": RoleDef(
        make_direct_readout_spec_,
        make_custom_reference_key("Direct Readout"),
        ro=Ro(
            EvalValue("$resonator.freq"), EvalValue("$resonator.wiring.ro_ch"), at=""
        ),
    ),
    # readout_dpm: the optimized pulse readout, seeded all-live from the
    # ro_optimize outputs (best_ro_*). The pulse window is best_ro_length + 0.1us,
    # the acquisition window is best_ro_length; missing calibration remains Invalid.
    "readout_dpm": RoleDef(
        make_pulse_readout_spec_,
        make_custom_reference_key("Pulse Readout"),
        (
            Pulse(
                EvalValue("$qubit.ext.best_ro_freq"),
                EvalValue("$resonator.wiring.ch"),
                EvalValue("$qubit.ext.best_ro_gain"),
                EvalValue("$qubit.ext.best_ro_length + 0.1"),
                at="pulse_cfg",
            ),
        ),
        ro=Ro(
            EvalValue("$qubit.ext.best_ro_freq"),
            EvalValue("$resonator.wiring.ro_ch"),
            EvalValue("$qubit.ext.best_ro_length"),
        ),
    ),
    # reset: "reset" is the library-aware pulse reset (ModuleInit.SMART prefers a
    # calibrated library entry; ModuleInit.INLINE seeds a fresh inline pulse reset — the
    # role formerly split out as "pulse_reset"). none / two_pulse / bath are the
    # other inline-only shapes.
    "reset": RoleDef(
        make_pulse_reset_spec_,
        make_custom_reference_key("Pulse Reset"),
        (Pulse(EvalValue("$qubit.freq"), _QUB_CH, 0.2, 1.0, at="pulse_cfg"),),
        lib=Lib(AbsResetCfg, ("$qubit.module.reset",)),
    ),
    "none_reset": RoleDef(
        make_none_reset_spec_, make_custom_reference_key("None Reset")
    ),
    "two_pulse_reset": RoleDef(
        make_two_pulse_reset_spec_,
        make_custom_reference_key("Two-Pulse Reset"),
        (
            Pulse(EvalValue("$qubit.freq"), _QUB_CH, 0.2, 1.0, at="pulse1_cfg"),
            Pulse(EvalValue("$qubit.freq"), _QUB_CH, 0.2, 1.0, at="pulse2_cfg"),
        ),
    ),
    "bath_reset": RoleDef(
        make_bath_reset_spec_,
        make_custom_reference_key("Bath Reset"),
        (
            Pulse(
                EvalValue("$resonator.freq"),
                EvalValue("$resonator.wiring.ch"),
                0.1,
                1.0,
                at="cavity_tone_cfg",
            ),
            Pulse(EvalValue("$qubit.freq"), _QUB_CH, 0.2, 1.0, at="qubit_tone_cfg"),
        ),
        refs=(Ref("pi2_cfg", "pi2_pulse"),),
    ),
    # waveforms
    "qub_waveform": RoleDef(
        make_cosine_waveform_spec_,
        make_custom_reference_key("Cosine"),
        wf_length=0.1,
        is_waveform=True,
    ),
    "res_waveform": RoleDef(
        make_const_waveform_spec_,
        make_custom_reference_key("Const"),
        wf_length=1.0,
        is_waveform=True,
    ),
}
