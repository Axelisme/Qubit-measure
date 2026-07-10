"""Unit tests for ``CfgBuilder`` — fluent value-tree assembly.

The builder seeds from the L1 blank tree, mounts L2 role factories at dotted
paths, sets scalar/sweep leaves, and Fast-Fails on bad path / kind / type. These
tests drive it in isolation with a MagicMock ctx (mirroring
``test_role_factories``) and real spec helpers (the builder descends specs to
validate ref kinds, so the spec must be real).
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_pulse_module_spec,
    make_readout_module_spec,
    make_reset_module_spec,
)
from zcu_tools.experiment.v2_gui.adapters.shared.cfg_builder import (
    CfgBuilder,
    RoleInit,
)
from zcu_tools.gui.app.main.adapter import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ExpContext,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    make_default_value,
)
from zcu_tools.gui.session.value_lookup import ValueKey, ValueRegistry
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import ModuleCfgFactory


def _empty_ctx() -> MagicMock:
    """ctx with empty md/ml — exercises blank fallback / library-miss paths."""
    ctx = MagicMock()
    ctx.md.get.side_effect = lambda k, d=None: d
    ctx.md.__contains__ = lambda self, k: False
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ctx.ml = ml
    return ctx


def _ctx_with_library_readout() -> ExpContext:
    ml = ModuleLibrary()
    ml.register_module(
        readout_rf=ModuleCfgFactory.from_raw(
            {
                "type": "readout/pulse",
                "pulse_cfg": {
                    "waveform": {"style": "const", "length": 1.0},
                    "ch": 1,
                    "nqz": 2,
                    "freq": 6100.0,
                    "gain": 0.2,
                },
                "ro_cfg": {
                    "ro_ch": 2,
                    "ro_freq": 6100.0,
                    "ro_length": 1.0,
                    "trig_offset": 0.5,
                },
            },
            ml=ml,
        )
    )
    return ExpContext(md=MetaDict(), ml=ml, soc=None, soccfg=None)


def _ctx_with_values(registry: ValueRegistry) -> ExpContext:
    return ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        values=registry,
    )


def _spec() -> CfgSectionSpec:
    """A twotone-freq-shaped spec: modules + top scalars + a freq sweep."""
    return CfgSectionSpec(
        fields={
            "modules": CfgSectionSpec(
                label="Modules",
                fields={
                    "reset": make_reset_module_spec(optional=True),
                    "qub_pulse": make_pulse_module_spec(),
                    "readout": make_readout_module_spec(),
                },
            ),
            "reps": ScalarSpec(label="Reps", type=int),
            "rounds": ScalarSpec(label="Rounds", type=int),
            "relax_delay": ScalarSpec(label="Relax delay (us)", type=float),
            "sweep": CfgSectionSpec(
                label="Sweep",
                fields={"freq": SweepSpec(label="Freq (MHz)")},
            ),
        }
    )


def _locked_spec() -> CfgSectionSpec:
    """Spec whose readout locks pulse_cfg.freq to 0.0 (mirrors onetone/freq)."""
    return CfgSectionSpec(
        fields={
            "modules": CfgSectionSpec(
                label="Modules",
                fields={
                    "readout": make_readout_module_spec().lock_literal(
                        "pulse_cfg.freq", 0.0
                    ),
                },
            ),
            "reps": ScalarSpec(label="Reps", type=int),
        }
    )


# --- structural completeness -------------------------------------------------


def test_build_with_no_overrides_equals_l1_blank():
    spec = _spec()
    built = CfgBuilder(_empty_ctx(), spec).build()
    blank = make_default_value(spec)
    assert set(built.fields) == set(blank.fields)
    # every top-level key present (ADR-0010 completeness)
    assert built.fields.keys() == {
        "modules",
        "reps",
        "rounds",
        "relax_delay",
        "sweep",
    }


# --- .scalars ----------------------------------------------------------------


def test_scalars_sets_top_level_leaves():
    b = CfgBuilder(_empty_ctx(), _spec()).scalars(reps=100, rounds=50, relax_delay=1.0)
    v = b.build()
    assert v.fields["reps"] == DirectValue(100)
    assert v.fields["rounds"] == DirectValue(50)
    assert v.fields["relax_delay"] == DirectValue(1.0)


def test_scalars_accepts_prewrapped_values():
    b = CfgBuilder(_empty_ctx(), _spec()).scalars(relax_delay=EvalValue(expr="t1"))
    v = b.build()
    rd = cast(Any, v.fields["relax_delay"])
    assert isinstance(rd, EvalValue) and rd.expr == "t1"


def test_scalars_unknown_key_fast_fails():
    with pytest.raises(RuntimeError, match="unknown top-level field 'nope'"):
        CfgBuilder(_empty_ctx(), _spec()).scalars(nope=1)


# --- .set --------------------------------------------------------------------


def test_set_overrides_scalar_inside_mounted_role():
    b = (
        CfgBuilder(_empty_ctx(), _spec())
        .role("modules.qub_pulse", "qub_probe")
        .set("modules.qub_pulse.gain", 0.05)
    )
    v = b.build()
    qub = cast(ReferenceValue, v.fields["modules"].fields["qub_pulse"])  # type: ignore[union-attr]
    assert qub.value.fields["gain"] == DirectValue(0.05)


def test_set_bad_path_fast_fails():
    with pytest.raises(KeyError, match="not found"):
        CfgBuilder(_empty_ctx(), _spec()).set("modules.no_such.gain", 1.0)


# --- .value_source ------------------------------------------------------------


def _device_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        fields={
            "dev": CfgSectionSpec(
                fields={
                    "flux_dev": ScalarSpec(
                        label="Flux Device",
                        type=str,
                        choices_source="devices",
                        required=True,
                    )
                }
            )
        }
    )


def test_value_source_resolves_device_ref_to_direct_value():
    registry = ValueRegistry()
    registry.register(
        ValueKey("device.flux.name", str),
        lambda: "flux",
        owner="test",
    )

    v = (
        CfgBuilder(_ctx_with_values(registry), _device_spec())
        .value_source("dev.flux_dev", "device.flux.name")
        .build()
    )

    dev = cast(CfgSectionValue, v.fields["dev"])
    assert dev.fields["flux_dev"] == DirectValue("flux")


def test_value_source_uses_default_when_source_is_missing():
    v = (
        CfgBuilder(_ctx_with_values(ValueRegistry()), _device_spec())
        .value_source("dev.flux_dev", "device.flux.name", default="flux_yoko")
        .build()
    )

    dev = cast(CfgSectionValue, v.fields["dev"])
    assert dev.fields["flux_dev"] == DirectValue("flux_yoko")


# --- .role -------------------------------------------------------------------


def test_role_mounts_module_ref():
    b = CfgBuilder(_empty_ctx(), _spec()).role("modules.qub_pulse", "qub_probe")
    v = b.build()
    node = v.fields["modules"].fields["qub_pulse"]  # type: ignore[union-attr]
    assert isinstance(node, ReferenceValue)


def test_role_adopt_uses_library_match():
    b = CfgBuilder(_ctx_with_library_readout(), _spec()).role(
        "modules.readout", "readout"
    )
    v = b.build()
    node = v.fields["modules"].fields["readout"]  # type: ignore[union-attr]
    assert isinstance(node, ReferenceValue)
    assert node.chosen_key == "readout_rf"


def test_role_disabled_library_miss_mounts_none():
    # empty ml + disabled optional reset → None, not a blank ref (ADR-0010)
    b = CfgBuilder(_empty_ctx(), _spec()).role(
        "modules.reset", "reset", RoleInit.DISABLED
    )
    v = b.build()
    assert v.fields["modules"].fields["reset"] is None  # type: ignore[union-attr]


def test_role_inline_forces_blank_even_when_library_has_match():
    b = CfgBuilder(_ctx_with_library_readout(), _spec()).role(
        "modules.readout", "readout", RoleInit.INLINE
    )
    v = b.build()
    node = v.fields["modules"].fields["readout"]  # type: ignore[union-attr]
    assert isinstance(node, ReferenceValue)
    assert node.chosen_key == "<Custom:Pulse Readout>"


def test_role_blank_overrides_custom_inline_value() -> None:
    value = (
        CfgBuilder(_ctx_with_library_readout(), _spec())
        .role(
            "modules.readout",
            "readout",
            RoleInit.INLINE,
            blank_overrides={"pulse_cfg.gain": 0.75},
        )
        .build()
    )

    modules = cast(CfgSectionValue, value.fields["modules"])
    readout = cast(ReferenceValue, modules.fields["readout"])
    pulse_cfg = cast(CfgSectionValue, readout.value.fields["pulse_cfg"])
    assert pulse_cfg.fields["gain"] == DirectValue(0.75)


def test_role_blank_overrides_adopt_fallback_blank() -> None:
    value = (
        CfgBuilder(_empty_ctx(), _spec())
        .role(
            "modules.readout",
            "readout",
            blank_overrides={"pulse_cfg.gain": 0.6},
        )
        .build()
    )

    modules = cast(CfgSectionValue, value.fields["modules"])
    readout = cast(ReferenceValue, modules.fields["readout"])
    pulse_cfg = cast(CfgSectionValue, readout.value.fields["pulse_cfg"])
    assert readout.chosen_key == "<Custom:Pulse Readout>"
    assert pulse_cfg.fields["gain"] == DirectValue(0.6)


def test_role_blank_overrides_ignore_linked_library_value() -> None:
    value = (
        CfgBuilder(_ctx_with_library_readout(), _spec())
        .role(
            "modules.readout",
            "readout",
            blank_overrides={"pulse_cfg.gain": 0.9},
        )
        .build()
    )

    modules = cast(CfgSectionValue, value.fields["modules"])
    readout = cast(ReferenceValue, modules.fields["readout"])
    pulse_cfg = cast(CfgSectionValue, readout.value.fields["pulse_cfg"])
    assert readout.chosen_key == "readout_rf"
    assert pulse_cfg.fields["gain"] == DirectValue(0.2)


def test_role_blank_overrides_reject_locked_path_without_mounting() -> None:
    builder = CfgBuilder(_empty_ctx(), _locked_spec())

    with pytest.raises(RuntimeError, match="locked literal"):
        builder.role(
            "modules.readout",
            "readout",
            RoleInit.INLINE,
            blank_overrides={"pulse_cfg.freq": 5.0},
        )

    modules = cast(CfgSectionValue, builder.build().fields["modules"])
    readout = cast(ReferenceValue, modules.fields["readout"])
    assert readout.chosen_key == "<Custom:Direct Readout>"


def test_role_blank_overrides_reject_unknown_path() -> None:
    with pytest.raises(KeyError, match="not found"):
        CfgBuilder(_empty_ctx(), _spec()).role(
            "modules.readout",
            "readout",
            RoleInit.INLINE,
            blank_overrides={"pulse_cfg.missing": 1.0},
        )


def test_role_blank_overrides_validate_all_before_mounting() -> None:
    builder = CfgBuilder(_empty_ctx(), _spec())

    with pytest.raises(KeyError, match="not found"):
        builder.role(
            "modules.readout",
            "readout",
            RoleInit.INLINE,
            blank_overrides={"pulse_cfg.gain": 0.8, "pulse_cfg.missing": 1.0},
        )

    modules = cast(CfgSectionValue, builder.build().fields["modules"])
    readout = cast(ReferenceValue, modules.fields["readout"])
    assert readout.chosen_key == "<Custom:Direct Readout>"


@pytest.mark.parametrize("relative_path", ["pulse_cfg", "pulse_cfg.waveform"])
def test_role_blank_overrides_reject_non_scalar_target_before_mounting(
    relative_path: str,
) -> None:
    builder = CfgBuilder(_empty_ctx(), _spec())

    with pytest.raises(RuntimeError, match="not a scalar leaf"):
        builder.role(
            "modules.readout",
            "readout",
            RoleInit.INLINE,
            blank_overrides={relative_path: 1.0},
        )

    modules = cast(CfgSectionValue, builder.build().fields["modules"])
    readout = cast(ReferenceValue, modules.fields["readout"])
    assert readout.chosen_key == "<Custom:Direct Readout>"


def test_role_blank_overrides_reject_sweep_target_before_mounting() -> None:
    spec = _spec()
    modules_spec = cast(CfgSectionSpec, spec.fields["modules"])
    readout_spec = cast(ReferenceSpec, modules_spec.fields["readout"])
    for allowed in readout_spec.allowed:
        allowed.fields["test_sweep"] = SweepSpec(label="Test sweep")
    builder = CfgBuilder(_empty_ctx(), spec)

    with pytest.raises(RuntimeError, match="not a scalar leaf"):
        builder.role(
            "modules.readout",
            "readout",
            RoleInit.INLINE,
            blank_overrides={"test_sweep": 1.0},
        )

    modules = cast(CfgSectionValue, builder.build().fields["modules"])
    readout = cast(ReferenceValue, modules.fields["readout"])
    assert readout.chosen_key == "<Custom:Direct Readout>"


def test_role_unknown_id_fast_fails():
    with pytest.raises(RuntimeError, match="unknown role_id 'bogus'"):
        CfgBuilder(_empty_ctx(), _spec()).role("modules.qub_pulse", "bogus")


def test_role_on_non_ref_spec_fast_fails():
    with pytest.raises(RuntimeError, match="not a ReferenceSpec"):
        CfgBuilder(_empty_ctx(), _spec()).role("reps", "qub_probe")


def test_role_rejects_waveform_role_for_module_spec_at_mount() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        CfgBuilder(_empty_ctx(), _spec()).role("modules.qub_pulse", "qub_waveform")

    assert str(exc_info.value) == (
        "CfgBuilder.role: spec at 'modules.qub_pulse' expects reference kind "
        "'module', but role 'qub_waveform' has kind 'waveform'"
    )


def test_role_rejects_module_role_for_waveform_spec_at_mount() -> None:
    spec = CfgSectionSpec(
        fields={
            "waveform": ReferenceSpec(
                kind="waveform",
                allowed=[CfgSectionSpec(label="Const", fields={})],
            )
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        CfgBuilder(_empty_ctx(), spec).role("waveform", "qub_probe")

    assert str(exc_info.value) == (
        "CfgBuilder.role: spec at 'waveform' expects reference kind 'waveform', "
        "but role 'qub_probe' has kind 'module'"
    )


def test_role_disabled_on_required_ref_fast_fails():
    # qub_pulse spec ref is NOT optional → RoleInit.DISABLED must raise
    with pytest.raises(RuntimeError, match="not optional"):
        CfgBuilder(_empty_ctx(), _spec()).role(
            "modules.qub_pulse", "qub_probe", RoleInit.DISABLED
        )


def test_role_disabled_blank_only_role_fast_fails():
    with pytest.raises(RuntimeError, match="has no library-aware"):
        CfgBuilder(_empty_ctx(), _spec()).role(
            "modules.reset", "none_reset", RoleInit.DISABLED
        )


def test_role_invalid_init_fast_fails():
    with pytest.raises(RuntimeError, match="init must be a RoleInit"):
        CfgBuilder(_empty_ctx(), _spec()).role(
            "modules.readout",
            "readout",
            "inline",  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        )


# --- .sweep -------------------------------------------------------------------


def test_sweep_mounts_explicit_value():
    sweep = SweepValue(start=0.0, stop=10.0, expts=11)
    b = CfgBuilder(_empty_ctx(), _spec()).sweep("sweep.freq", sweep)
    v = b.build()
    sw = cast(SweepValue, v.fields["sweep"].fields["freq"])  # type: ignore[union-attr]
    assert sw.start == 0.0 and sw.stop == 10.0 and sw.expts == 11
    assert sw.step == pytest.approx(1.0)  # (10-0)/(11-1)


def test_sweep_accepts_eval_edges():
    sw = SweepValue(
        start=EvalValue(expr="q_f - 20"), stop=EvalValue(expr="q_f + 20"), expts=301
    )
    b = CfgBuilder(_empty_ctx(), _spec()).sweep("sweep.freq", sw)
    v = b.build()
    mounted = v.fields["sweep"].fields["freq"]  # type: ignore[union-attr]
    assert mounted is sw


def test_sweep_on_non_sweep_spec_fast_fails():
    with pytest.raises(RuntimeError, match="not a SweepSpec"):
        CfgBuilder(_empty_ctx(), _spec()).sweep(
            "reps", SweepValue(start=0.0, stop=1.0, expts=11)
        )


# --- locked literals (auto-fill + reject .set) -------------------------------


def test_build_fills_locked_literal_inside_mounted_ref():
    # .role mounts an L2 readout whose pulse_cfg.freq carries the md value;
    # build() must overwrite it with the spec's locked 0.0.
    b = CfgBuilder(_empty_ctx(), _locked_spec()).role(
        "modules.readout", "readout", RoleInit.INLINE
    )
    v = b.build()
    readout = cast(ReferenceValue, v.fields["modules"].fields["readout"])  # type: ignore[union-attr]
    pulse_cfg = cast(Any, readout.value.fields["pulse_cfg"])
    assert pulse_cfg.fields["freq"] == DirectValue(0.0)


def test_build_fills_top_level_locked_literal():
    spec = CfgSectionSpec(fields={"reps": LiteralSpec(value=1, label="Reps")})
    v = CfgBuilder(_empty_ctx(), spec).build()
    assert v.fields["reps"] == DirectValue(1)


def test_set_on_locked_path_fast_fails():
    with pytest.raises(RuntimeError, match="locked literal"):
        CfgBuilder(_empty_ctx(), _locked_spec()).role(
            "modules.readout", "readout", RoleInit.INLINE
        ).set("modules.readout.pulse_cfg.freq", 5.0)


def test_set_rejects_inconsistent_allowed_shape_leaf_types() -> None:
    spec = CfgSectionSpec(
        fields={
            "module": ReferenceSpec(
                kind="module",
                allowed=[
                    CfgSectionSpec(
                        label="Writable",
                        fields={"value": ScalarSpec(label="Value", type=float)},
                    ),
                    CfgSectionSpec(
                        label="Locked",
                        fields={"value": LiteralSpec(value=0.0)},
                    ),
                ],
            )
        }
    )

    with pytest.raises(TypeError, match="inconsistent spec types"):
        CfgBuilder(_empty_ctx(), spec).set("module.value", 1.0)


# --- one-shot ----------------------------------------------------------------


def test_build_is_one_shot():
    b = CfgBuilder(_empty_ctx(), _spec())
    b.build()
    with pytest.raises(RuntimeError, match="already built"):
        b.scalars(reps=1)
