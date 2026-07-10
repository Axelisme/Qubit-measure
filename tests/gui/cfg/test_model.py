"""Tests for spec fluent overrides (lock_literal) + value with_field."""

from __future__ import annotations

from typing import Any, cast

import pytest
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    WaveformRefSpec,
)


def _readout_pulse_freq(spec: CfgSectionSpec) -> Any:
    """Dig to allowed[0].pulse_cfg.freq (test navigation, cast past the union)."""
    ref = cast(Any, spec.fields["modules"]).fields["readout"]
    return ref.allowed[0].fields["pulse_cfg"].fields["freq"]


def _nested_spec() -> CfgSectionSpec:
    shape_a = CfgSectionSpec(
        label="A",
        fields={
            "pulse_cfg": CfgSectionSpec(fields={"freq": ScalarSpec("Freq", float)})
        },
    )
    shape_b = CfgSectionSpec(
        label="B",
        fields={"ro_cfg": CfgSectionSpec(fields={"ro_freq": ScalarSpec("RO", float)})},
    )
    return CfgSectionSpec(
        fields={
            "modules": CfgSectionSpec(
                fields={"readout": ModuleRefSpec(allowed=[shape_a, shape_b])}
            ),
            "reps": ScalarSpec("Reps", int),
        }
    )


# --- spec.lock_literal ------------------------------------------------------


def test_lock_literal_replaces_leaf_with_literal_spec():
    spec = _nested_spec()
    locked = spec.lock_literal("modules.readout.pulse_cfg.freq", 0.0)
    leaf = _readout_pulse_freq(locked)
    assert isinstance(leaf, LiteralSpec)
    assert leaf.value == 0.0


def test_lock_literal_returns_new_frozen_spec_original_untouched():
    spec = _nested_spec()
    spec.lock_literal("modules.readout.pulse_cfg.freq", 0.0)
    orig = _readout_pulse_freq(spec)
    assert isinstance(orig, ScalarSpec)  # original tree unchanged (frozen)


def test_lock_literal_duck_type_skips_allowed_without_path():
    spec = _nested_spec()
    locked = spec.lock_literal("modules.readout.pulse_cfg.freq", 0.0)
    shape_b = cast(Any, locked.fields["modules"]).fields["readout"].allowed[1]
    # shape B has no pulse_cfg → untouched (same object identity preserved)
    orig_b = cast(Any, spec.fields["modules"]).fields["readout"].allowed[1]
    assert shape_b is orig_b


def test_lock_literal_chains():
    spec = _nested_spec()
    # chaining works because each call returns a new CfgSectionSpec
    locked = spec.lock_literal("modules.readout.pulse_cfg.freq", 0.0).lock_literal(
        "reps", 1
    )
    leaf = _readout_pulse_freq(locked)
    assert isinstance(leaf, LiteralSpec)
    assert isinstance(locked.fields["reps"], LiteralSpec)


def test_module_ref_spec_lock_literal_is_chain_start():
    """ModuleRefSpec.lock_literal locks a leaf relative to its allowed shapes,
    so a sub-tree from a helper can be locked as it is built (path is shorter,
    no need to start from the root section). Returns a ModuleRefSpec for chaining."""
    inner = CfgSectionSpec(
        label="A",
        fields={
            "pulse_cfg": CfgSectionSpec(fields={"freq": ScalarSpec("Freq", float)})
        },
    )
    ref = ModuleRefSpec(allowed=[inner])
    locked = ref.lock_literal("pulse_cfg.freq", 0.0)
    assert isinstance(locked, ModuleRefSpec)
    leaf = cast(Any, locked.allowed[0]).fields["pulse_cfg"].fields["freq"]
    assert isinstance(leaf, LiteralSpec)
    assert leaf.value == 0.0
    # original untouched (frozen)
    orig = cast(Any, ref.allowed[0]).fields["pulse_cfg"].fields["freq"]
    assert isinstance(orig, ScalarSpec)


def test_lock_literal_raises_when_no_allowed_matches():
    spec = _nested_spec()
    with pytest.raises(RuntimeError, match="not found in any allowed"):
        spec.lock_literal("modules.readout.nonexistent.x", 0.0)


def test_lock_literal_raises_on_unknown_top_segment():
    spec = _nested_spec()
    with pytest.raises(RuntimeError, match="not found"):
        spec.lock_literal("nope.freq", 0.0)


# --- lock_literal descent through WaveformRefSpec (framework symmetry) -------


def _pulse_with_waveform() -> CfgSectionSpec:
    """A pulse shape whose 'waveform' is a WaveformRefSpec with two shapes."""
    const = CfgSectionSpec(label="Const", fields={"length": ScalarSpec("Len", float)})
    gauss = CfgSectionSpec(
        label="Gauss",
        fields={
            "length": ScalarSpec("Len", float),
            "sigma": ScalarSpec("Sigma", float),
        },
    )
    return CfgSectionSpec(
        label="Pulse",
        fields={
            "freq": ScalarSpec("Freq", float),
            "waveform": WaveformRefSpec(allowed=[const, gauss]),
        },
    )


def test_lock_literal_descends_into_waveform_ref_via_module_ref():
    """A leaf nested inside a WaveformRefSpec can be locked from the parent
    ModuleRefSpec — symmetric with the ModuleRefSpec descent."""
    ref = ModuleRefSpec(allowed=[_pulse_with_waveform()])
    locked = ref.lock_literal("waveform.length", 0.0)
    wf = cast(Any, locked.allowed[0]).fields["waveform"]
    assert isinstance(wf, WaveformRefSpec)
    # every allowed waveform shape with a 'length' leaf is locked (duck-typed)
    for shape in wf.allowed:
        leaf = shape.fields["length"]
        assert isinstance(leaf, LiteralSpec)
        assert leaf.value == 0.0
    # a sibling leaf (gauss.sigma) stays editable
    assert isinstance(wf.allowed[1].fields["sigma"], ScalarSpec)


def test_waveform_ref_spec_lock_literal_is_chain_start():
    """WaveformRefSpec.lock_literal returns a WaveformRefSpec; original frozen."""
    const = CfgSectionSpec(label="Const", fields={"length": ScalarSpec("Len", float)})
    ref = WaveformRefSpec(allowed=[const])
    locked = ref.lock_literal("length", 0.0)
    assert isinstance(locked, WaveformRefSpec)
    assert isinstance(locked.allowed[0].fields["length"], LiteralSpec)
    assert isinstance(ref.allowed[0].fields["length"], ScalarSpec)  # untouched


def test_lock_literal_descends_waveform_ref_via_full_section_path():
    """End-to-end: section → module ref → pulse → waveform ref → leaf."""
    spec = CfgSectionSpec(
        fields={
            "modules": CfgSectionSpec(
                fields={"qub_pulse": ModuleRefSpec(allowed=[_pulse_with_waveform()])}
            )
        }
    )
    locked = spec.lock_literal("modules.qub_pulse.waveform.length", 0.0)
    wf = (
        cast(Any, locked.fields["modules"])
        .fields["qub_pulse"]
        .allowed[0]
        .fields["waveform"]
    )
    assert all(isinstance(s.fields["length"], LiteralSpec) for s in wf.allowed)


def test_lock_literal_waveform_ref_raises_when_no_shape_matches():
    const = CfgSectionSpec(label="Const", fields={"length": ScalarSpec("Len", float)})
    ref = WaveformRefSpec(allowed=[const])
    with pytest.raises(RuntimeError, match="not found in any allowed"):
        ref.lock_literal("nope", 0.0)


# --- value.with_field -------------------------------------------------------


def _nested_value() -> CfgSectionValue:
    inner = CfgSectionValue(
        fields={
            "pulse_cfg": CfgSectionValue(
                fields={"gain": DirectValue(0.1), "freq": DirectValue(0.0)}
            )
        }
    )
    return CfgSectionValue(
        fields={
            "readout": ModuleRefValue(chosen_key="<Custom:X>", value=inner),
            "reps": DirectValue(100),
        }
    )


def test_with_field_sets_scalar_in_place_and_returns_self():
    val = _nested_value()
    out = val.with_field("readout.pulse_cfg.gain", 0.05)
    assert out is val  # in-place, returns self
    assert (
        cast(Any, val.fields["readout"]).value.fields["pulse_cfg"].fields["gain"].value
        == 0.05
    )


def test_with_field_chains():
    val = _nested_value()
    val.with_field("readout.pulse_cfg.gain", 0.05).with_field("reps", 200)
    assert cast(Any, val.fields["reps"]).value == 200


def test_with_field_accepts_prebuilt_eval_value():
    val = _nested_value()
    val.with_field("readout.pulse_cfg.freq", EvalValue("q_f", 4000.0))
    leaf = cast(Any, val.fields["readout"]).value.fields["pulse_cfg"].fields["freq"]
    assert isinstance(leaf, EvalValue)
    assert leaf.expr == "q_f"


def test_with_field_module_ref_with_field_delegates():
    inner = CfgSectionValue(fields={"gain": DirectValue(0.1)})
    ref = ModuleRefValue(chosen_key="<Custom:X>", value=inner)
    out = ref.with_field("gain", 0.3)
    assert out is ref
    assert cast(Any, ref.value.fields["gain"]).value == 0.3


def test_with_field_raises_on_bad_descent():
    val = _nested_value()
    with pytest.raises(RuntimeError, match="cannot descend"):
        val.with_field("reps.x", 1)  # reps is a scalar, can't descend


def test_with_field_empty_path_raises():
    val = _nested_value()
    with pytest.raises(RuntimeError, match="must not be empty"):
        val.with_field("", 1)
