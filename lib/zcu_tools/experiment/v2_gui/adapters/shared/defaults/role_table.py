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

The seed value carriers (:class:`Md` / :class:`Source` / :data:`TRIG`) wrap the
existing md-mechanism and the read-only value-source escape hatch. They resolve
against the whole ``ExpContext`` (predictor / device / project / …) while still
materializing to direct/eval scalar leaves before the cfg tree leaves the builder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.adapter import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ModuleRefValue,
    ScalarValue,
    WaveformRefValue,
)
from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec as make_pulse_spec_
from zcu_tools.gui.app.main.specs.readout import (
    make_direct_readout_spec as make_direct_readout_spec_,
)
from zcu_tools.gui.app.main.specs.readout import (
    make_pulse_readout_spec as make_pulse_readout_spec_,
)
from zcu_tools.gui.app.main.specs.reset import (
    make_bath_reset_spec as make_bath_reset_spec_,
)
from zcu_tools.gui.app.main.specs.reset import (
    make_none_reset_spec as make_none_reset_spec_,
)
from zcu_tools.gui.app.main.specs.reset import (
    make_pulse_reset_spec as make_pulse_reset_spec_,
)
from zcu_tools.gui.app.main.specs.reset import (
    make_two_pulse_reset_spec as make_two_pulse_reset_spec_,
)
from zcu_tools.gui.app.main.specs.waveform import (
    make_const_waveform_spec as make_const_waveform_spec_,
)
from zcu_tools.gui.app.main.specs.waveform import (
    make_cosine_waveform_spec as make_cosine_waveform_spec_,
)
from zcu_tools.gui.session.value_lookup import ValueRef, resolve_value_ref
from zcu_tools.program.v2.modules import AbsResetCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.pulse import PulseCfg

from ..ctx_helpers import md_has_key
from .helpers import (
    make_default_value,
    make_trig_offset,
    patch_pulse_fields,
    patch_ro_cfg_fields,
    select_named_module_value,
    select_named_waveform_value,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import CfgSectionSpec, ExpContext

_RefNode = ModuleRefValue | WaveformRefValue

# ---------------------------------------------------------------------------
# Seed value carriers — the md-mechanism today, a ValueSource seam tomorrow.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Md:
    """A md-linked seed gated on ``key``: a live ``EvalValue`` (of ``expr`` when
    given — for arithmetic like ``"best_ro_length + 0.1"`` — else of the bare
    ``key``) when md has ``key``; otherwise ``default`` (a constant whose int/float
    type is preserved, or a nested ``Md`` fallback chain)."""

    key: str
    default: float | int | Md = 0
    expr: str | None = None


@dataclass(frozen=True)
class Source:
    """A read-only value-source seed resolved once while building a role default."""

    key: str
    type_name: str | None = None


class _Trig(Enum):
    TRIG = "trig"


#: The readout trig-offset rule: ``EvalValue("timeFly + 0.05")`` when md has
#: ``timeFly``, else ``DirectValue(0.55)`` (see ``make_trig_offset``).
TRIG = _Trig.TRIG

#: A seed value in a RoleDef: a raw scalar, an md-linked ``Md``, a registered
#: ``Source``, or the ``TRIG`` rule (ro_cfg only).
SeedVal = float | int | str | bool | Md | Source | _Trig


def _resolve(ctx: ExpContext, v: SeedVal) -> ScalarValue:
    """Lower a seed value to a scalar leaf (DirectValue / EvalValue)."""
    if isinstance(v, Md):
        if md_has_key(ctx, v.key):
            return EvalValue(expr=v.expr if v.expr is not None else v.key)
        return _resolve(ctx, v.default)
    if isinstance(v, Source):
        return DirectValue(resolve_value_ref(ValueRef(v.key, v.type_name), ctx.values))
    if v is TRIG:
        return make_trig_offset(ctx, trig_expr="timeFly + 0.05", trig_fallback=0.55)
    return DirectValue(v)


# ---------------------------------------------------------------------------
# Role description — pure data.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lib:
    """A library adopt policy: a module type (``None`` → waveform library) and the
    preferred entry names in priority order (load-bearing)."""

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
    if not isinstance(sub, CfgSectionValue):
        raise RuntimeError(f"role seed path {at!r} is not a section")
    return sub


def _adopt_waveform(value: CfgSectionValue, ctx: ExpContext, name: str) -> None:
    """Adopt a library waveform into the pulse's ``waveform`` sub-ref when present
    (mirrors the readout factories' ro_waveform block)."""
    from zcu_tools.gui.app.main.cfg_schemas import waveform_cfg_to_value

    pulse_cfg = value.fields.get("pulse_cfg")
    if not isinstance(pulse_cfg, CfgSectionValue):
        return
    waveform_ref = pulse_cfg.fields.get("waveform")
    if isinstance(waveform_ref, WaveformRefValue) and name in ctx.ml.waveforms:
        _, wav_val = waveform_cfg_to_value(ctx.ml.waveforms[name])
        pulse_cfg.fields["waveform"] = WaveformRefValue(chosen_key=name, value=wav_val)


def role_blank(role: RoleDef, ctx: ExpContext) -> _RefNode:
    """Assemble a role's blank value tree from its data (never a library lookup)."""
    value = make_default_value(role.shape())

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

    if role.adopt_waveform is not None:
        _adopt_waveform(value, ctx, role.adopt_waveform)

    if role.is_waveform:
        return WaveformRefValue(chosen_key=role.tag, value=value)
    return ModuleRefValue(role.tag, value)


def role_ref(
    role: RoleDef, ctx: ExpContext, *, optional: bool = False
) -> _RefNode | None:
    """Library-aware mount: adopt the first preferred-named library entry, else the
    blank (or ``None`` when ``optional`` and nothing matches, ADR-0010)."""
    lib = role.lib
    if lib is None:
        raise RuntimeError(f"role {role.tag!r} has no library policy (ref unsupported)")
    if lib.module_type is None:
        selected = select_named_waveform_value(ctx.ml, list(lib.prefer))
        if selected is not None:
            return selected
    else:
        selected = select_named_module_value(
            ml=ctx.ml, module_type=lib.module_type, preferred_names=list(lib.prefer)
        )
        if selected is not None:
            return ModuleRefValue(chosen_key=selected.name, value=selected.value)
    if optional:
        return None
    return role_blank(role, ctx)


# ---------------------------------------------------------------------------
# ROLE_TABLE — the single declarative source of every role default.
# ---------------------------------------------------------------------------

# Shared seed: a blank qubit pulse (q_f / qub_ch). qub_probe, pi_pulse and
# pi2_pulse all default to this and differ only in their library names.
_QUB_PULSE: tuple[Pulse, ...] = (Pulse(Md("q_f", 4000.0), Md("qub_ch", 0), 0.1, 5.1),)

ROLE_TABLE: dict[str, RoleDef] = {
    # qubit / resonator probe pulses
    "qub_probe": RoleDef(
        make_pulse_spec_,
        "<Custom:Pulse>",
        _QUB_PULSE,
        lib=Lib(PulseCfg, ("qub_probe",)),
    ),
    "res_probe": RoleDef(
        make_pulse_spec_,
        "<Custom:Pulse>",
        (Pulse(Md("r_f", 6000.0), Md("res_ch", 0), 0.05, 1.0),),
        lib=Lib(PulseCfg, ("res_probe",)),
    ),
    "pi_pulse": RoleDef(
        make_pulse_spec_,
        "<Custom:Pulse>",
        _QUB_PULSE,
        lib=Lib(PulseCfg, ("pi_amp", "pi_len")),
    ),
    "pi2_pulse": RoleDef(
        make_pulse_spec_,
        "<Custom:Pulse>",
        _QUB_PULSE,
        lib=Lib(PulseCfg, ("pi2_amp", "pi2_len", "pi_amp", "pi_len")),
    ),
    # readout: "readout" is the library-aware pulse readout (Init.ADOPT prefers a
    # calibrated library entry; Init.INLINE seeds a fresh inline pulse readout —
    # the role formerly split out as "pulse_readout"). direct_readout / readout_dpm
    # are the other inline-only shapes.
    "readout": RoleDef(
        make_pulse_readout_spec_,
        "<Custom:Pulse Readout>",
        (Pulse(Md("r_f", 6000.0), Md("res_ch", 0), 0.1, 1.0, at="pulse_cfg"),),
        ro=Ro(Md("r_f", 6000.0), Md("ro_ch", 0)),
        adopt_waveform="ro_waveform",
        lib=Lib(
            PulseReadoutCfg, ("readout_dpm", "readout_rf", "readout", "res_readout")
        ),
    ),
    "direct_readout": RoleDef(
        make_direct_readout_spec_,
        "<Custom:Direct Readout>",
        ro=Ro(Md("r_f", 6000.0), Md("ro_ch", 0), at=""),
    ),
    # readout_dpm: the optimized pulse readout, seeded all-live from the
    # ro_optimize outputs (best_ro_*). The pulse window is best_ro_length + 0.1us,
    # the acquisition window is best_ro_length; freq falls back to r_f then 6000.
    "readout_dpm": RoleDef(
        make_pulse_readout_spec_,
        "<Custom:Pulse Readout>",
        (
            Pulse(
                Md("best_ro_freq", Md("r_f", 6000.0)),
                Md("res_ch", 0),
                Md("best_ro_gain", 0.1),
                Md("best_ro_length", 1.1, expr="best_ro_length + 0.1"),
                at="pulse_cfg",
            ),
        ),
        ro=Ro(
            Md("best_ro_freq", Md("r_f", 6000.0)),
            Md("ro_ch", 0),
            Md("best_ro_length", 1.0),
        ),
        adopt_waveform="ro_waveform",
    ),
    # reset: "reset" is the library-aware pulse reset (Init.ADOPT prefers a
    # calibrated library entry; Init.INLINE seeds a fresh inline pulse reset — the
    # role formerly split out as "pulse_reset"). none / two_pulse / bath are the
    # other inline-only shapes.
    "reset": RoleDef(
        make_pulse_reset_spec_,
        "<Custom:Pulse Reset>",
        (Pulse(Md("q_f", 4000.0), Md("qub_ch", 0), 0.2, 1.0, at="pulse_cfg"),),
        lib=Lib(AbsResetCfg, ("reset_bath", "reset_10", "reset_120")),
    ),
    "none_reset": RoleDef(make_none_reset_spec_, "<Custom:None Reset>"),
    "two_pulse_reset": RoleDef(
        make_two_pulse_reset_spec_,
        "<Custom:Two-Pulse Reset>",
        (
            Pulse(Md("q_f", 4000.0), Md("qub_ch", 0), 0.2, 1.0, at="pulse1_cfg"),
            Pulse(Md("q_f", 4000.0), Md("qub_ch", 0), 0.2, 1.0, at="pulse2_cfg"),
        ),
    ),
    "bath_reset": RoleDef(
        make_bath_reset_spec_,
        "<Custom:Bath Reset>",
        (
            Pulse(Md("r_f", 6000.0), Md("res_ch", 0), 0.1, 1.0, at="cavity_tone_cfg"),
            Pulse(Md("q_f", 4000.0), Md("qub_ch", 0), 0.2, 1.0, at="qubit_tone_cfg"),
            Pulse(Md("q_f", 4000.0), Md("qub_ch", 0), 0.2, 1.0, at="pi2_cfg"),
        ),
    ),
    # waveforms
    "qub_waveform": RoleDef(
        make_cosine_waveform_spec_,
        "<Custom:Cosine>",
        wf_length=0.1,
        is_waveform=True,
        lib=Lib(None, ("qub_flat", "qub_cos")),
    ),
    "res_waveform": RoleDef(
        make_const_waveform_spec_,
        "<Custom:Const>",
        wf_length=1.0,
        is_waveform=True,
        lib=Lib(None, ("res_flat", "res_const")),
    ),
}
