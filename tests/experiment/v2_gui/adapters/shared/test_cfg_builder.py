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
from zcu_tools.experiment.v2_gui.adapters.shared.cfg_builder import CfgBuilder
from zcu_tools.gui.app.main.adapter import (
    CfgSectionSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    make_default_value,
)


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


# --- .role -------------------------------------------------------------------


def test_role_mounts_module_ref():
    b = CfgBuilder(_empty_ctx(), _spec()).role("modules.qub_pulse", "qub_probe")
    v = b.build()
    node = v.fields["modules"].fields["qub_pulse"]  # type: ignore[union-attr]
    assert isinstance(node, ModuleRefValue)


def test_role_optional_library_miss_mounts_none():
    # empty ml + optional reset → disabled (None), not a blank ref (ADR-0010)
    b = CfgBuilder(_empty_ctx(), _spec()).role("modules.reset", "reset", optional=True)
    v = b.build()
    assert v.fields["modules"].fields["reset"] is None  # type: ignore[union-attr]


def test_role_prefer_blank_forces_inline():
    # readout role has a ref factory, but prefer_blank forces the inline blank
    b = CfgBuilder(_empty_ctx(), _spec()).role(
        "modules.readout", "readout", prefer_blank=True
    )
    v = b.build()
    node = v.fields["modules"].fields["readout"]  # type: ignore[union-attr]
    assert isinstance(node, ModuleRefValue)
    assert node.chosen_key.startswith("<Custom:")  # inline, not a library name


def test_role_unknown_id_fast_fails():
    with pytest.raises(RuntimeError, match="unknown role_id 'bogus'"):
        CfgBuilder(_empty_ctx(), _spec()).role("modules.qub_pulse", "bogus")


def test_role_on_non_ref_spec_fast_fails():
    with pytest.raises(RuntimeError, match="not a ModuleRefSpec/WaveformRefSpec"):
        CfgBuilder(_empty_ctx(), _spec()).role("reps", "qub_probe")


def test_role_optional_on_required_ref_fast_fails():
    # qub_pulse spec ref is NOT optional → optional=True must raise
    with pytest.raises(RuntimeError, match="not optional"):
        CfgBuilder(_empty_ctx(), _spec()).role(
            "modules.qub_pulse", "qub_probe", optional=True
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
            EvalValue(expr="q_f"),
            10.0,
            11,  # type: ignore[arg-type]
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
        "modules.readout", "pulse_readout"
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
            "modules.readout", "pulse_readout"
        ).set("modules.readout.pulse_cfg.freq", 5.0)


# --- one-shot ----------------------------------------------------------------


def test_build_is_one_shot():
    b = CfgBuilder(_empty_ctx(), _spec())
    b.build()
    with pytest.raises(RuntimeError, match="already built"):
        b.scalars(reps=1)
