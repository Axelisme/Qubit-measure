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
from zcu_tools.experiment.v2_gui.adapters.shared.cfg_builder import CfgBuilder, Init
from zcu_tools.gui.app.main.adapter import (
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    ExpContext,
    LiteralSpec,
    ModuleRefValue,
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
    qub = cast(ModuleRefValue, v.fields["modules"].fields["qub_pulse"])  # type: ignore[union-attr]
    assert qub.value.fields["gain"] == DirectValue(0.05)


def test_set_bad_path_fast_fails():
    with pytest.raises(RuntimeError, match="does not resolve"):
        CfgBuilder(_empty_ctx(), _spec()).set("modules.no_such.gain", 1.0)


# --- .value_ref ---------------------------------------------------------------


def _device_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        fields={
            "dev": CfgSectionSpec(
                fields={"flux_dev": DeviceRefSpec(label="Flux Device")}
            )
        }
    )


def test_value_ref_resolves_device_ref_to_direct_value():
    registry = ValueRegistry()
    registry.register(
        ValueKey("device.flux.name", str),
        lambda: "flux",
        owner="test",
    )

    v = (
        CfgBuilder(_ctx_with_values(registry), _device_spec())
        .value_ref("dev.flux_dev", "device.flux.name")
        .build()
    )

    dev = cast(CfgSectionValue, v.fields["dev"])
    assert dev.fields["flux_dev"] == DirectValue("flux")


def test_value_ref_uses_default_when_source_is_missing():
    v = (
        CfgBuilder(_ctx_with_values(ValueRegistry()), _device_spec())
        .value_ref("dev.flux_dev", "device.flux.name", default="flux_yoko")
        .build()
    )

    dev = cast(CfgSectionValue, v.fields["dev"])
    assert dev.fields["flux_dev"] == DirectValue("flux_yoko")


# --- .role -------------------------------------------------------------------


def test_role_mounts_module_ref():
    b = CfgBuilder(_empty_ctx(), _spec()).role("modules.qub_pulse", "qub_probe")
    v = b.build()
    node = v.fields["modules"].fields["qub_pulse"]  # type: ignore[union-attr]
    assert isinstance(node, ModuleRefValue)


def test_role_adopt_uses_library_match():
    b = CfgBuilder(_ctx_with_library_readout(), _spec()).role(
        "modules.readout", "readout"
    )
    v = b.build()
    node = v.fields["modules"].fields["readout"]  # type: ignore[union-attr]
    assert isinstance(node, ModuleRefValue)
    assert node.chosen_key == "readout_rf"


def test_role_disabled_library_miss_mounts_none():
    # empty ml + disabled optional reset → None, not a blank ref (ADR-0010)
    b = CfgBuilder(_empty_ctx(), _spec()).role("modules.reset", "reset", Init.DISABLED)
    v = b.build()
    assert v.fields["modules"].fields["reset"] is None  # type: ignore[union-attr]


def test_role_inline_forces_blank_even_when_library_has_match():
    b = CfgBuilder(_ctx_with_library_readout(), _spec()).role(
        "modules.readout", "readout", Init.INLINE
    )
    v = b.build()
    node = v.fields["modules"].fields["readout"]  # type: ignore[union-attr]
    assert isinstance(node, ModuleRefValue)
    assert node.chosen_key == "<Custom:Pulse Readout>"


def test_role_unknown_id_fast_fails():
    with pytest.raises(RuntimeError, match="unknown role_id 'bogus'"):
        CfgBuilder(_empty_ctx(), _spec()).role("modules.qub_pulse", "bogus")


def test_role_on_non_ref_spec_fast_fails():
    with pytest.raises(RuntimeError, match="not a ModuleRefSpec/WaveformRefSpec"):
        CfgBuilder(_empty_ctx(), _spec()).role("reps", "qub_probe")


def test_role_disabled_on_required_ref_fast_fails():
    # qub_pulse spec ref is NOT optional → Init.DISABLED must raise
    with pytest.raises(RuntimeError, match="not optional"):
        CfgBuilder(_empty_ctx(), _spec()).role(
            "modules.qub_pulse", "qub_probe", Init.DISABLED
        )


def test_role_disabled_blank_only_role_fast_fails():
    with pytest.raises(RuntimeError, match="has no library-aware"):
        CfgBuilder(_empty_ctx(), _spec()).role(
            "modules.reset", "none_reset", Init.DISABLED
        )


def test_role_invalid_init_fast_fails():
    with pytest.raises(RuntimeError, match="init must be an Init"):
        CfgBuilder(_empty_ctx(), _spec()).role(
            "modules.readout",
            "readout",
            "inline",  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        )


# --- .sweep / .set_sweep -----------------------------------------------------


def test_sweep_literal_math():
    b = CfgBuilder(_empty_ctx(), _spec()).sweep("sweep.freq", 0.0, 10.0, 11)
    v = b.build()
    sw = cast(SweepValue, v.fields["sweep"].fields["freq"])  # type: ignore[union-attr]
    assert sw.start == 0.0 and sw.stop == 10.0 and sw.expts == 11
    assert sw.step == pytest.approx(1.0)  # (10-0)/(11-1)


def test_sweep_rejects_eval_edge():
    with pytest.raises(RuntimeError, match="EvalValue edge requires set_sweep"):
        CfgBuilder(_empty_ctx(), _spec()).sweep(
            "sweep.freq",
            EvalValue(expr="q_f"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            10.0,
            11,
        )


def test_set_sweep_accepts_eval_edges():
    sw = SweepValue(
        start=EvalValue(expr="q_f - 20"), stop=EvalValue(expr="q_f + 20"), expts=301
    )
    b = CfgBuilder(_empty_ctx(), _spec()).set_sweep("sweep.freq", sw)
    v = b.build()
    mounted = v.fields["sweep"].fields["freq"]  # type: ignore[union-attr]
    assert mounted is sw


def test_set_sweep_on_non_sweep_spec_fast_fails():
    with pytest.raises(RuntimeError, match="not a SweepSpec"):
        CfgBuilder(_empty_ctx(), _spec()).set_sweep(
            "reps", SweepValue(start=0.0, stop=1.0, expts=11)
        )


# --- locked literals (auto-fill + reject .set) -------------------------------


def test_build_fills_locked_literal_inside_mounted_ref():
    # .role mounts an L2 readout whose pulse_cfg.freq carries the md value;
    # build() must overwrite it with the spec's locked 0.0.
    b = CfgBuilder(_empty_ctx(), _locked_spec()).role(
        "modules.readout", "readout", Init.INLINE
    )
    v = b.build()
    readout = cast(ModuleRefValue, v.fields["modules"].fields["readout"])  # type: ignore[union-attr]
    pulse_cfg = cast(Any, readout.value.fields["pulse_cfg"])
    assert pulse_cfg.fields["freq"] == DirectValue(0.0)


def test_build_fills_top_level_locked_literal():
    spec = CfgSectionSpec(fields={"reps": LiteralSpec(value=1, label="Reps")})
    v = CfgBuilder(_empty_ctx(), spec).build()
    assert v.fields["reps"] == DirectValue(1)


def test_set_on_locked_path_fast_fails():
    with pytest.raises(RuntimeError, match="locked literal"):
        CfgBuilder(_empty_ctx(), _locked_spec()).role(
            "modules.readout", "readout", Init.INLINE
        ).set("modules.readout.pulse_cfg.freq", 5.0)


# --- one-shot ----------------------------------------------------------------


def test_build_is_one_shot():
    b = CfgBuilder(_empty_ctx(), _spec())
    b.build()
    with pytest.raises(RuntimeError, match="already built"):
        b.scalars(reps=1)
