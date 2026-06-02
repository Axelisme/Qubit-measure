"""Tests for the session cfg codec (raw↔live), WorkspaceService's internal
capture/apply implementation. Disk I/O + payload-shape validation now live in
the PersistenceCaretaker / pydantic memento (test_caretaker / test_persistence_
types); this file covers only the cfg lowering / rebuild transforms."""

from __future__ import annotations

import pytest
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    DisabledRefValue,
    EvalValue,
    ModuleRefSpec,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
)
from zcu_tools.gui.services.session_codec import (
    SessionCodecError,
    raw_to_schema,
    schema_to_raw,
)


def _empty(spec: CfgSectionSpec) -> CfgSchema:
    return CfgSchema(spec=spec, value=CfgSectionValue(fields={}))


def test_rejects_legacy_sweep_step_none():
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"sweep": SweepSpec(label="Sweep")}),
        value=CfgSectionValue(fields={}),
    )
    with pytest.raises(SessionCodecError, match="Sweep step is required"):
        raw_to_schema(
            base,
            {"sweep": {"start": 0.0, "stop": 1.0, "expts": 11, "step": None}},
        )


def test_rejects_legacy_sweep_eval_edges():
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"sweep": SweepSpec(label="Sweep")}),
        value=CfgSectionValue(fields={}),
    )
    with pytest.raises(SessionCodecError, match="Legacy sweep"):
        raw_to_schema(
            base,
            {
                "sweep": {
                    "start": "=r_f - 10",
                    "stop": "=r_f + 10",
                    "expts": 11,
                    "step": 2.0,
                }
            },
        )


def test_roundtrip_preserves_eval_values():
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "freq": ScalarSpec(label="Freq", type=float),
                "sweep": SweepSpec(label="Sweep"),
            }
        ),
        value=CfgSectionValue(
            fields={
                "freq": EvalValue(expr="r_f", resolved=6000.0, error=None),
                "sweep": SweepValue(
                    start=EvalValue(expr="r_f - rf_w", resolved=5980.0, error=None),
                    stop=EvalValue(expr="r_f + rf_w", resolved=6020.0, error=None),
                    expts=101,
                    step=0.4,
                ),
            }
        ),
    )

    restored = raw_to_schema(_empty(schema.spec), schema_to_raw(schema))

    freq = restored.value.fields["freq"]
    sweep = restored.value.fields["sweep"]
    assert isinstance(freq, EvalValue)
    assert freq.expr == "r_f"
    assert isinstance(sweep, SweepValue)
    assert isinstance(sweep.start, EvalValue)
    assert isinstance(sweep.stop, EvalValue)
    assert sweep.start.expr == "r_f - rf_w"
    assert sweep.stop.expr == "r_f + rf_w"


def test_waveform_ref_roundtrip():
    inner_spec = CfgSectionSpec(
        fields={"width": ScalarSpec(label="Width", type=float)},
        label="Gaussian",
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={"wf": WaveformRefSpec(allowed=[inner_spec], label="Waveform")}
        ),
        value=CfgSectionValue(
            fields={
                "wf": WaveformRefValue(
                    chosen_key="Gaussian",
                    value=CfgSectionValue(fields={"width": DirectValue(value=50.0)}),
                ),
            }
        ),
    )

    restored = raw_to_schema(_empty(schema.spec), schema_to_raw(schema))

    wf = restored.value.fields["wf"]
    assert isinstance(wf, WaveformRefValue)
    assert wf.chosen_key == "Gaussian"
    assert wf.value.fields["width"] == DirectValue(value=50.0)
    assert wf.is_overridden is False


def test_disabled_module_ref_roundtrip():
    """A disabled optional ModuleRef (DisabledRefValue, ADR-0012) survives."""
    inner_spec = CfgSectionSpec(
        fields={"gain": ScalarSpec(label="Gain", type=float)},
        label="Pulse",
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "reset": ModuleRefSpec(
                    allowed=[inner_spec], label="Reset", optional=True
                ),
            }
        ),
        value=CfgSectionValue(fields={"reset": DisabledRefValue()}),
    )

    raw = schema_to_raw(schema)
    assert raw["reset"] == {"__kind": "disabled"}
    restored = raw_to_schema(_empty(schema.spec), raw)
    assert isinstance(restored.value.fields["reset"], DisabledRefValue)


def test_disabled_waveform_ref_roundtrip():
    inner_spec = CfgSectionSpec(
        fields={"width": ScalarSpec(label="Width", type=float)},
        label="Gaussian",
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "wf": WaveformRefSpec(
                    allowed=[inner_spec], label="Waveform", optional=True
                ),
            }
        ),
        value=CfgSectionValue(fields={"wf": DisabledRefValue()}),
    )

    raw = schema_to_raw(schema)
    assert raw["wf"] == {"__kind": "disabled"}
    restored = raw_to_schema(_empty(schema.spec), raw)
    assert isinstance(restored.value.fields["wf"], DisabledRefValue)


def test_waveform_ref_preserves_override():
    inner_spec = CfgSectionSpec(
        fields={"width": ScalarSpec(label="Width", type=float)},
        label="Gaussian",
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={"wf": WaveformRefSpec(allowed=[inner_spec], label="Waveform")}
        ),
        value=CfgSectionValue(
            fields={
                "wf": WaveformRefValue(
                    chosen_key="Gaussian",
                    value=CfgSectionValue(fields={"width": DirectValue(value=50.0)}),
                    is_overridden=True,
                ),
            }
        ),
    )

    restored = raw_to_schema(_empty(schema.spec), schema_to_raw(schema))
    wf = restored.value.fields["wf"]
    assert isinstance(wf, WaveformRefValue)
    assert wf.is_overridden is True


def test_rejects_legacy_scalar_eval_expr():
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"freq": ScalarSpec(label="Freq", type=float)}),
        value=CfgSectionValue(fields={}),
    )
    with pytest.raises(SessionCodecError, match="Legacy scalar"):
        raw_to_schema(base, {"freq": "=r_f + 10"})


def test_device_ref_value_must_be_string():
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"dev": DeviceRefSpec(label="Device")}),
        value=CfgSectionValue(fields={}),
    )
    with pytest.raises(
        SessionCodecError, match="Device reference value must be string"
    ):
        raw_to_schema(
            base,
            {"dev": {"__kind": "direct", "value": 123, "is_unset": False}},
        )


def test_device_ref_must_use_direct_encoding():
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"dev": DeviceRefSpec(label="Device")}),
        value=CfgSectionValue(fields={}),
    )
    with pytest.raises(SessionCodecError, match="Device reference must use direct"):
        raw_to_schema(base, {"dev": "lo_device"})
