"""Unit tests for dynamic value-tree validation through finished lowering.

Validates the *dynamic* contract (every scalar has a value, every EvalValue
resolves against md, every device ref is selected). The *static* contract
(structure, LiteralSpec, type/choices) is in test_validate.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import DeviceRefSpec


def _ml() -> MagicMock:
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    return ml


def _md(**attrs: object) -> MagicMock:
    md = MagicMock()
    for k, v in attrs.items():
        setattr(md, k, v)
    return md


def _inner_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Pulse",
        fields={"type": LiteralSpec("pulse"), "gain": ScalarSpec("Gain", float)},
    )


# --- scalar: unset / has value -----------------------------------------------


def test_scalar_unset_raises():
    spec = CfgSectionSpec(fields={"reps": ScalarSpec("Reps", int)})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"reps": DirectValue(None)})
    )
    with pytest.raises(RuntimeError, match="is unset"):
        schema_to_raw_dict(schema, _md(), _ml())


def test_scalar_with_value_passes():
    spec = CfgSectionSpec(fields={"reps": ScalarSpec("Reps", int)})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"reps": DirectValue(5)})
    )
    schema_to_raw_dict(schema, _md(), _ml())


# --- EvalValue resolve -------------------------------------------------------


def test_eval_resolvable_passes():
    spec = CfgSectionSpec(fields={"freq": ScalarSpec("Freq", float)})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"freq": EvalValue(expr="r_f")})
    )
    schema_to_raw_dict(schema, _md(r_f=6.0), _ml())


def test_eval_unresolvable_raises():
    spec = CfgSectionSpec(fields={"freq": ScalarSpec("Freq", float)})
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(fields={"freq": EvalValue(expr="missing_key")}),
    )
    with pytest.raises(RuntimeError, match="failed"):
        schema_to_raw_dict(schema, _md(), _ml())


def test_eval_type_mismatch_raises():
    spec = CfgSectionSpec(fields={"ch": ScalarSpec("Channel", int)})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"ch": EvalValue(expr="r_f")})
    )
    with pytest.raises(RuntimeError, match="failed"):
        schema_to_raw_dict(schema, _md(r_f=6.5), _ml())


# --- SweepValue with EvalValue edges -----------------------------------------


def test_sweep_eval_edge_passes():
    spec = CfgSectionSpec(fields={"s": SweepSpec(label="S")})
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={"s": SweepValue(start=EvalValue(expr="x"), stop=1.0, expts=11)}
        ),
    )
    schema_to_raw_dict(schema, _md(x=0.0), _ml())


def test_sweep_eval_edge_fails():
    spec = CfgSectionSpec(fields={"s": SweepSpec(label="S")})
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={
                "s": SweepValue(start=EvalValue(expr="missing"), stop=1.0, expts=11)
            }
        ),
    )
    with pytest.raises(RuntimeError, match="failed"):
        schema_to_raw_dict(schema, _md(), _ml())


def test_centered_sweep_eval_center_passes():
    spec = CfgSectionSpec(fields={"s": CenteredSweepSpec(label="S")})
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={
                "s": CenteredSweepValue(center=EvalValue(expr="x"), span=1.0, expts=11)
            }
        ),
    )
    schema_to_raw_dict(schema, _md(x=0.0), _ml())


def test_centered_sweep_eval_center_fails():
    spec = CfgSectionSpec(fields={"s": CenteredSweepSpec(label="S")})
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={
                "s": CenteredSweepValue(
                    center=EvalValue(expr="missing"),
                    span=1.0,
                    expts=11,
                )
            }
        ),
    )
    with pytest.raises(RuntimeError, match="failed"):
        schema_to_raw_dict(schema, _md(), _ml())


# --- DeviceRefSpec ------------------------------------------------------------


def test_device_empty_raises():
    spec = CfgSectionSpec(fields={"dev": DeviceRefSpec(label="Flux Device")})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"dev": DirectValue("")})
    )
    with pytest.raises(RuntimeError, match="device not selected"):
        schema_to_raw_dict(schema, _md(), _ml())


def test_device_none_raises():
    spec = CfgSectionSpec(fields={"dev": DeviceRefSpec(label="Flux Device")})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"dev": DirectValue(None)})
    )
    with pytest.raises(RuntimeError, match="device not selected"):
        schema_to_raw_dict(schema, _md(), _ml())


def test_device_selected_passes():
    spec = CfgSectionSpec(fields={"dev": DeviceRefSpec(label="Flux Device")})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"dev": DirectValue("YOKO_1")})
    )
    schema_to_raw_dict(schema, _md(), _ml())


# --- refs (disabled / enabled recursion) --------------------------------------


def test_disabled_optional_ref_skipped():
    spec = CfgSectionSpec(
        fields={"m": ModuleRefSpec(allowed=[_inner_spec()], optional=True)}
    )
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"m": None}))
    schema_to_raw_dict(schema, _md(), _ml())


def test_enabled_ref_recurses_into_unset_scalar():
    inner = _inner_spec()
    spec = CfgSectionSpec(fields={"m": ModuleRefSpec(allowed=[inner], label="M")})
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={
                "m": ModuleRefValue(
                    "<Custom:Pulse>",
                    CfgSectionValue(
                        fields={
                            "type": DirectValue("pulse"),
                            "gain": DirectValue(None),
                        }
                    ),
                )
            }
        ),
    )
    with pytest.raises(RuntimeError, match="m.gain.*is unset"):
        schema_to_raw_dict(schema, _md(), _ml())


# --- LiteralSpec (skipped by dynamic) ----------------------------------------


def test_literal_skipped():
    spec = CfgSectionSpec(fields={"t": LiteralSpec(value=1)})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"t": DirectValue(1)}))
    schema_to_raw_dict(schema, _md(), _ml())


# --- nested CfgSectionSpec ---------------------------------------------------


def test_nested_section_recurses():
    spec = CfgSectionSpec(
        fields={
            "sub": CfgSectionSpec(fields={"x": ScalarSpec("X", int)}),
        }
    )
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={"sub": CfgSectionValue(fields={"x": DirectValue(None)})}
        ),
    )
    with pytest.raises(RuntimeError, match="sub.x.*is unset"):
        schema_to_raw_dict(schema, _md(), _ml())


# --- integration with to_raw_dict --------------------------------------------


def test_to_raw_dict_calls_dynamic_before_lowering():
    spec = CfgSectionSpec(fields={"reps": ScalarSpec("Reps", int)})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"reps": DirectValue(None)})
    )
    with pytest.raises(RuntimeError, match="is unset") as exc_info:
        schema_to_raw_dict(schema, _md(), _ml())
    # validate_dynamic message includes "(no value to lower)"
    assert "no value to lower" in str(exc_info.value)


def test_to_raw_dict_skips_dynamic_when_no_md():
    spec = CfgSectionSpec(fields={"freq": ScalarSpec("Freq", float)})
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(fields={"freq": EvalValue(expr="r_f", resolved=6.0)}),
    )
    # md=None → dynamic validate skipped, lowering uses snapshot
    result = schema_to_raw_dict(schema, None, _ml())
    assert result["freq"] == 6.0
