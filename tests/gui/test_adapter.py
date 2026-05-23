"""Unit tests for zcu_tools.gui.adapter (Phase 19 — Spec/Value split)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.adapter import (
    AnalyzeParam,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ModuleRefSpec,
    ModuleRefValue,
    MultiSweepSpec,
    MultiSweepValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    analyze_params_to_raw_dict,
    make_default_value,
    schema_to_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ml() -> MagicMock:
    ml = MagicMock()
    return ml


def _schema(spec_fields: dict, val_fields: dict | None = None) -> CfgSchema:
    spec = CfgSectionSpec(fields=spec_fields)
    value = CfgSectionValue(fields=val_fields or {})
    return CfgSchema(spec=spec, value=value)


# ---------------------------------------------------------------------------
# ScalarSpec / ScalarValue
# ---------------------------------------------------------------------------


def test_scalar_int():
    s = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(100)},
    )
    assert schema_to_dict(s, _make_ml()) == {"reps": 100}


def test_scalar_str():
    s = _schema(
        {"name": ScalarSpec(label="Name", type=str)},
        {"name": DirectValue("hello")},
    )
    assert schema_to_dict(s, _make_ml())["name"] == "hello"


def test_scalar_editable_false_still_included():
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float, editable=False)},
        {"freq": DirectValue(6.0)},
    )
    assert schema_to_dict(s, _make_ml())["freq"] == pytest.approx(6.0)


def test_scalar_eval_value_uses_resolved_snapshot():
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="r_f - rf_w", resolved=5998.0)},
    )
    assert schema_to_dict(s, _make_ml())["freq"] == pytest.approx(5998.0)


def test_scalar_eval_value_unresolved_raises():
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="missing_name", resolved=None)},
    )
    with pytest.raises(RuntimeError, match="freq.*missing_name.*unresolved"):
        schema_to_dict(s, _make_ml())


def test_scalar_missing_in_value_skipped():
    """A key present in spec but absent in value raises."""
    s = _schema(
        {
            "reps": ScalarSpec(label="Reps", type=int),
            "x": ScalarSpec(label="X", type=int),
        },
        {"reps": DirectValue(5)},
    )
    with pytest.raises(RuntimeError, match="x"):
        schema_to_dict(s, _make_ml())


def test_extra_value_fields_raise():
    s = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(5), "extra": DirectValue(1)},
    )
    with pytest.raises(RuntimeError, match="extra"):
        schema_to_dict(s, _make_ml())


# ---------------------------------------------------------------------------
# SweepSpec / SweepValue
# ---------------------------------------------------------------------------


def test_sweep_produces_sweep_cfg():
    from zcu_tools.program.v2 import SweepCfg

    s = _schema(
        {"sweep": SweepSpec(label="Freq")},
        {"sweep": SweepValue(start=1.0, stop=2.0, expts=11)},
    )
    result = schema_to_dict(s, _make_ml())
    sweep = result["sweep"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.start == pytest.approx(1.0)
    assert sweep.stop == pytest.approx(2.0)
    assert sweep.expts == 11


def test_sweep_step_mode():
    from zcu_tools.program.v2 import SweepCfg

    s = _schema(
        {"sweep": SweepSpec()},
        {"sweep": SweepValue(start=0.0, stop=1.0, expts=0, step=0.1)},
    )
    result = schema_to_dict(s, _make_ml())
    assert isinstance(result["sweep"], SweepCfg)


# ---------------------------------------------------------------------------
# MultiSweepSpec / MultiSweepValue
# ---------------------------------------------------------------------------


def test_multi_sweep_produces_dict_of_sweeps():
    from zcu_tools.program.v2 import SweepCfg

    s = _schema(
        {
            "sweep": MultiSweepSpec(
                axes={
                    "freq": SweepSpec(label="Freq"),
                    "gain": SweepSpec(label="Gain"),
                }
            )
        },
        {
            "sweep": MultiSweepValue(
                axes={
                    "freq": SweepValue(5.0, 6.0, 5),
                    "gain": SweepValue(0.0, 1.0, 3),
                }
            )
        },
    )
    result = schema_to_dict(s, _make_ml())
    assert set(result["sweep"].keys()) == {"freq", "gain"}
    assert isinstance(result["sweep"]["freq"], SweepCfg)
    assert result["sweep"]["freq"].expts == 5
    assert result["sweep"]["gain"].expts == 3


# ---------------------------------------------------------------------------
# ModuleRefSpec / ModuleRefValue — value is directly flattened (no ml.get_module)
# ---------------------------------------------------------------------------


def test_module_ref_named_key_uses_value_tree():
    """schema_to_dict always expands the value tree, regardless of Named vs Custom."""
    ml = _make_ml()
    inner_spec = CfgSectionSpec(
        label="Direct Readout",
        fields={"ro_ch": ScalarSpec(label="RO ch", type=int)},
    )
    s = _schema(
        {"readout": ModuleRefSpec(allowed=[inner_spec])},
        {
            "readout": ModuleRefValue(
                chosen_key="readout_rf",
                value=CfgSectionValue(fields={"ro_ch": DirectValue(99)}),
            )
        },
    )
    result = schema_to_dict(s, ml)
    # value tree (99) takes precedence — ml is not consulted for dict output
    assert result["readout"] == {"ro_ch": 99}


def test_module_ref_custom_key_resolves_by_label():
    inner_spec = CfgSectionSpec(
        label="Direct Readout",
        fields={"gain": ScalarSpec(label="Gain", type=float)},
    )
    s = _schema(
        {"ro": ModuleRefSpec(allowed=[inner_spec])},
        {
            "ro": ModuleRefValue(
                chosen_key="<Custom:Direct Readout>",
                value=CfgSectionValue(fields={"gain": DirectValue(0.5)}),
            )
        },
    )
    result = schema_to_dict(s, _make_ml())
    assert result["ro"] == {"gain": pytest.approx(0.5)}


# ---------------------------------------------------------------------------
# CfgSectionSpec nesting
# ---------------------------------------------------------------------------


def test_nested_section_is_recursed():
    s = _schema(
        {"inner": CfgSectionSpec(fields={"x": ScalarSpec(label="X", type=int)})},
        {"inner": CfgSectionValue(fields={"x": DirectValue(42)})},
    )
    assert schema_to_dict(s, _make_ml()) == {"inner": {"x": 42}}


# ---------------------------------------------------------------------------
# make_default_value
# ---------------------------------------------------------------------------


def test_make_default_value_scalar():
    spec = CfgSectionSpec(fields={"x": ScalarSpec(label="X", type=int)})
    val = make_default_value(spec)
    assert isinstance(val.fields["x"], DirectValue)


def test_analyze_params_to_raw_dict_round_trips_scalars():
    params = [
        AnalyzeParam(key="threshold", label="Threshold", type=float, default=0.5),
        AnalyzeParam(
            key="model_type",
            label="Model type",
            type=str,
            default="hm",
            choices=["hm", "t", "auto"],
        ),
    ]
    raw = analyze_params_to_raw_dict(
        params,
        {"threshold": 0.75, "model_type": "auto"},
    )
    assert raw == {"threshold": 0.75, "model_type": "auto"}


def test_analyze_params_to_raw_dict_rejects_missing_key():
    params = [AnalyzeParam(key="threshold", label="Threshold", type=float, default=0.5)]
    with pytest.raises(RuntimeError, match="Missing analyze params"):
        analyze_params_to_raw_dict(params, {})


def test_analyze_params_to_raw_dict_rejects_unknown_key():
    params = [AnalyzeParam(key="threshold", label="Threshold", type=float, default=0.5)]
    with pytest.raises(RuntimeError, match="Unknown analyze params"):
        analyze_params_to_raw_dict(params, {"threshold": 0.5, "extra": 1})


def test_make_default_value_sweep():
    spec = CfgSectionSpec(fields={"s": SweepSpec()})
    val = make_default_value(spec)
    sv = val.fields["s"]
    assert isinstance(sv, SweepValue)
    assert sv.expts == 11


def test_make_default_value_nested():
    inner = CfgSectionSpec(fields={"y": ScalarSpec(label="Y", type=float)})
    spec = CfgSectionSpec(fields={"sub": inner})
    val = make_default_value(spec)
    sub = val.fields["sub"]
    assert isinstance(sub, CfgSectionValue)
    assert isinstance(sub.fields["y"], DirectValue)


def test_make_default_value_module_ref():
    inner_spec = CfgSectionSpec(
        label="Direct Readout",
        fields={"ro_ch": ScalarSpec(label="RO ch", type=int)},
    )
    spec = CfgSectionSpec(fields={"ro": ModuleRefSpec(allowed=[inner_spec])})
    val = make_default_value(spec)
    ro = val.fields["ro"]
    assert isinstance(ro, ModuleRefValue)
    assert ro.chosen_key == "<Custom:Direct Readout>"
    assert isinstance(ro.value, CfgSectionValue)
