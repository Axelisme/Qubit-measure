"""Tests for the session cfg codec (raw↔live), WorkspaceService's internal
capture/apply implementation. Disk I/O + payload-shape validation now live in
the PersistenceCaretaker / pydantic memento (test_caretaker / test_persistence_
types); this file covers only the cfg lowering / rebuild transforms."""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
)
from zcu_tools.gui.app.main.services.session_codec import (
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
    """A disabled optional ModuleRef is ``None`` in the value tree (ADR-0010)
    and survives capture/restore as None — not re-enabled to the first allowed."""
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
        value=CfgSectionValue(fields={"reset": None}),
    )

    raw = schema_to_raw(schema)
    assert raw["reset"] == {"__kind": "disabled"}
    restored = raw_to_schema(_empty(schema.spec), raw)
    assert restored.value.fields["reset"] is None


def test_disabled_module_ref_missing_key_restores_to_none():
    """A key absent from the persisted payload (old file / never stored) for an
    optional ref restores to None (disabled), not the enabled allowed[0] —
    this is the lookback persist bug (ADR-0010)."""
    inner_spec = CfgSectionSpec(
        fields={"gain": ScalarSpec(label="Gain", type=float)},
        label="Pulse",
    )
    spec = CfgSectionSpec(
        fields={
            "reset": ModuleRefSpec(allowed=[inner_spec], label="Reset", optional=True),
        }
    )
    restored = raw_to_schema(CfgSchema(spec=spec, value=_empty(spec).value), {})
    assert restored.value.fields["reset"] is None


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
        value=CfgSectionValue(fields={"wf": None}),
    )

    raw = schema_to_raw(schema)
    assert raw["wf"] == {"__kind": "disabled"}
    restored = raw_to_schema(_empty(schema.spec), raw)
    assert restored.value.fields["wf"] is None


def test_enabled_module_ref_roundtrip():
    """A non-disabled optional ModuleRef survives as an enabled ModuleRefValue."""
    inner_spec = CfgSectionSpec(
        fields={"gain": ScalarSpec(label="Gain", type=float)},
        label="Pulse",
    )
    spec = CfgSectionSpec(
        fields={
            "reset": ModuleRefSpec(allowed=[inner_spec], label="Reset", optional=True),
        }
    )
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={
                "reset": ModuleRefValue(
                    "<Custom:Pulse>",
                    CfgSectionValue(fields={"gain": DirectValue(0.3)}),
                )
            }
        ),
    )
    restored = raw_to_schema(_empty(spec), schema_to_raw(schema))
    reset = restored.value.fields["reset"]
    assert isinstance(reset, ModuleRefValue)
    assert reset.value.fields["gain"] == DirectValue(0.3)


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
            {"dev": {"__kind": "direct", "value": 123}},
        )


def test_device_ref_must_use_direct_encoding():
    base = CfgSchema(
        spec=CfgSectionSpec(fields={"dev": DeviceRefSpec(label="Device")}),
        value=CfgSectionValue(fields={}),
    )
    with pytest.raises(SessionCodecError, match="Device reference must use direct"):
        raw_to_schema(base, {"dev": "lo_device"})


def test_complete_literal_key_round_trips():
    """A complete value tree (the LiteralSpec key has an entry) round-trips: the
    literal serialises as its value, not a disabled marker (only optional refs
    use ``disabled``)."""
    spec = CfgSectionSpec(
        fields={
            "reps": LiteralSpec(value=1, label="Reps"),
            "rounds": ScalarSpec(label="Rounds", type=int),
        }
    )
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={"reps": DirectValue(1), "rounds": DirectValue(500)}
        ),
    )

    raw = schema_to_raw(schema)
    assert raw["reps"] == {"__kind": "direct", "value": 1}  # not {"__kind":"disabled"}

    restored = raw_to_schema(_empty(spec), raw)
    reps = restored.value.fields["reps"]
    assert isinstance(reps, DirectValue) and reps.value == 1


def test_incomplete_value_tree_fails_fast_on_capture():
    """A value tree that omits a non-ref key is an invariant violation — capture
    fast-fails rather than silently filling a default (value trees are complete
    by contract; only optional refs may be a None entry)."""
    spec = CfgSectionSpec(
        fields={
            "reps": LiteralSpec(value=1, label="Reps"),
            "rounds": ScalarSpec(label="Rounds", type=int),
        }
    )
    # omits the LiteralSpec 'reps' key → incomplete
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"rounds": DirectValue(500)})
    )
    with pytest.raises(SessionCodecError, match="incomplete value tree"):
        schema_to_raw(schema)
