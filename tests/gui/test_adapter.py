"""Unit tests for zcu_tools.gui.adapter (Phase 2 — no Qt, no hardware)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSection,
    ModuleRefField,
    MultiSweepField,
    ScalarField,
    SweepField,
    schema_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ml() -> MagicMock:
    """Return a minimal ModuleLibrary mock."""
    ml = MagicMock()
    ml.get_module.side_effect = lambda name, override=None: {
        "name": name,
        "override": override,
    }
    return ml


def _schema(fields: dict) -> CfgSchema:
    return CfgSchema(root=CfgSection(fields=fields))


# ---------------------------------------------------------------------------
# ScalarField
# ---------------------------------------------------------------------------


def test_scalar_field_int():
    schema = _schema({"reps": ScalarField(value=100, label="Reps", type=int)})
    result = schema_to_dict(schema, _make_ml())
    assert result == {"reps": 100}


def test_scalar_field_str():
    schema = _schema({"name": ScalarField(value="hello", label="Name", type=str)})
    result = schema_to_dict(schema, _make_ml())
    assert result["name"] == "hello"


def test_scalar_field_editable_false_does_not_affect_value():
    schema = _schema(
        {"freq": ScalarField(value=6.0, label="Freq", type=float, editable=False)}
    )
    result = schema_to_dict(schema, _make_ml())
    assert result["freq"] == 6.0


# ---------------------------------------------------------------------------
# SweepField
# ---------------------------------------------------------------------------


def test_sweep_field_produces_sweep_cfg():
    from zcu_tools.program.v2 import SweepCfg

    schema = _schema({"sweep": SweepField(start=1.0, stop=2.0, expts=11)})
    result = schema_to_dict(schema, _make_ml())
    sweep = result["sweep"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.start == pytest.approx(1.0)
    assert sweep.stop == pytest.approx(2.0)
    assert sweep.expts == 11


# ---------------------------------------------------------------------------
# MultiSweepField
# ---------------------------------------------------------------------------


def test_multi_sweep_field_produces_dict_of_sweeps():
    from zcu_tools.program.v2 import SweepCfg

    schema = _schema(
        {
            "sweep": MultiSweepField(
                sweeps={
                    "freq": SweepField(start=5.0, stop=6.0, expts=5),
                    "gain": SweepField(start=0.0, stop=1.0, expts=3),
                }
            )
        }
    )
    result = schema_to_dict(schema, _make_ml())
    assert set(result["sweep"].keys()) == {"freq", "gain"}
    assert isinstance(result["sweep"]["freq"], SweepCfg)
    assert result["sweep"]["freq"].expts == 5
    assert result["sweep"]["gain"].expts == 3


# ---------------------------------------------------------------------------
# ModuleRefField — named module path
# ---------------------------------------------------------------------------


def test_module_ref_field_named_calls_get_module():
    ml = _make_ml()
    schema = _schema(
        {
            "readout": ModuleRefField(
                module_name="ro_pulse",
                override={"gain": 0.1},
                inline_cfg=None,
                expanded_content=None,
                available_modules=["ro_pulse"],
            )
        }
    )
    result = schema_to_dict(schema, ml)
    ml.get_module.assert_called_once_with("ro_pulse", {"gain": 0.1})
    assert result["readout"]["name"] == "ro_pulse"


def test_module_ref_field_named_empty_override_passes_none():
    """Empty override dict should be treated as None (no override)."""
    ml = _make_ml()
    schema = _schema(
        {
            "readout": ModuleRefField(
                module_name="ro_pulse",
                override={},
                inline_cfg=None,
                expanded_content=None,
                available_modules=["ro_pulse"],
            )
        }
    )
    schema_to_dict(schema, ml)
    ml.get_module.assert_called_once_with("ro_pulse", None)


# ---------------------------------------------------------------------------
# ModuleRefField — inline path
# ---------------------------------------------------------------------------


def test_module_ref_field_inline_cfg_returns_inline_dict():
    ml = _make_ml()
    inline = {"type": "custom", "gain": 0.5}
    schema = _schema(
        {
            "readout": ModuleRefField(
                module_name=None,
                override={},
                inline_cfg=inline,
                expanded_content=None,
                available_modules=[],
            )
        }
    )
    result = schema_to_dict(schema, ml)
    assert result["readout"] == inline
    ml.get_module.assert_not_called()


def test_module_ref_field_inline_none_returns_empty_dict():
    schema = _schema(
        {
            "readout": ModuleRefField(
                module_name=None,
                override={},
                inline_cfg=None,
                expanded_content=None,
                available_modules=[],
            )
        }
    )
    result = schema_to_dict(schema, _make_ml())
    assert result["readout"] == {}


# ---------------------------------------------------------------------------
# CfgSection nesting
# ---------------------------------------------------------------------------


def test_nested_section_is_recursed():
    schema = _schema(
        {
            "inner": CfgSection(
                fields={
                    "x": ScalarField(value=42, label="X", type=int),
                }
            )
        }
    )
    result = schema_to_dict(schema, _make_ml())
    assert result == {"inner": {"x": 42}}


# ---------------------------------------------------------------------------
# Mixed schema
# ---------------------------------------------------------------------------


def test_mixed_schema():
    schema = _schema(
        {
            "reps": ScalarField(value=100, label="Reps", type=int),
            "sweep": SweepField(start=0.0, stop=1.0, expts=3),
            "cfg": CfgSection(
                fields={"gain": ScalarField(value=0.05, label="Gain", type=float)}
            ),
        }
    )
    result = schema_to_dict(schema, _make_ml())
    assert result["reps"] == 100
    assert result["sweep"].expts == 3
    assert result["cfg"]["gain"] == pytest.approx(0.05)
