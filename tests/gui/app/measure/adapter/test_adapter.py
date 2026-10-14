"""Unit tests for zcu_tools.gui.app.measure.adapter."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ModuleWriteback,
    NoAnalysisResult,
    ParameterWriteback,
    WaveformWriteback,
    require_soc_handles,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.measure.adapter.protocol import NoAnalyzeParams
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    make_default_value,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.app.measure._components import QubitSchema, ResonatorSchema, WiringSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _schema(spec_fields: dict, val_fields: dict | None = None) -> CfgSchema:
    spec = CfgSectionSpec(fields=spec_fields)
    value = CfgSectionValue(fields=val_fields or {})
    return CfgSchema(spec=spec, value=value)


def _empty_cfg_definition() -> MeasureCfgDefinition:
    return MeasureCfgBuilder(role_factories={}).build()


def test_require_soc_handles_is_framework_request_validation() -> None:
    from zcu_tools.gui.app.measure.adapter import RunRequest

    with pytest.raises(RuntimeError, match="soc is required"):
        require_soc_handles(RunRequest(soc=None, soccfg=MagicMock(), cfg_context=None))


# ---------------------------------------------------------------------------
# ScalarSpec / ScalarValue
# ---------------------------------------------------------------------------


def test_scalar_int():
    s = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(100)},
    )
    assert schema_to_raw_dict(s, None) == {"reps": 100}


def test_scalar_str():
    s = _schema(
        {"name": ScalarSpec(label="Name", type=str)},
        {"name": DirectValue("hello")},
    )
    assert schema_to_raw_dict(s, None)["name"] == "hello"


def test_scalar_editable_false_still_included():
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float, editable=False)},
        {"freq": DirectValue(6.0)},
    )
    assert schema_to_raw_dict(s, None)["freq"] == pytest.approx(6.0)


def test_scalar_eval_value_uses_resolved_snapshot():
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="R1.freq - R1.kappa", resolved=5998.0)},
    )
    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(kind="resonator", freq=6000.0, kappa=2.0),
            "Q1": QubitSchema(kind="qubit/transmon"),
        }
    )
    assert schema_to_raw_dict(s, parameters)["freq"] == pytest.approx(5998.0)


def test_scalar_eval_value_unresolved_raises():
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="missing_name", resolved=None)},
    )
    with pytest.raises(RuntimeError) as error:
        schema_to_raw_dict(s, None)
    assert "missing_name" in str(error.value)
    assert "freq" in str(error.value)


def test_scalar_eval_value_resolves_against_md_when_no_snapshot():
    """An EvalValue built without a snapshot is resolved by lowering against md."""
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="R1.freq - R1.kappa")},
    )
    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(kind="resonator", freq=6000.0, kappa=2.0),
            "Q1": QubitSchema(kind="qubit/transmon"),
        }
    )
    assert schema_to_raw_dict(s, parameters)["freq"] == pytest.approx(5998.0)


def test_scalar_eval_snapshot_drift_warns_but_keeps_snapshot(caplog):
    """When md is supplied and the snapshot disagrees with a fresh evaluation,
    lowering keeps the snapshot but logs a drift warning (commit cross-check)."""
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="R1.freq - R1.kappa", resolved=5998.0)},
    )
    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(kind="resonator", freq=7000.0, kappa=2.0),
            "Q1": QubitSchema(kind="qubit/transmon"),
        }
    )
    with caplog.at_level("WARNING"):
        out = schema_to_raw_dict(s, parameters)
    # snapshot still wins
    assert out["freq"] == pytest.approx(5998.0)
    assert any(
        "differs from current md evaluation" in r.message for r in caplog.records
    )


def test_scalar_eval_snapshot_consistent_no_warn(caplog):
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="R1.freq - R1.kappa", resolved=5998.0)},
    )
    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(kind="resonator", freq=6000.0, kappa=2.0),
            "Q1": QubitSchema(kind="qubit/transmon"),
        }
    )
    with caplog.at_level("WARNING"):
        schema_to_raw_dict(s, parameters)
    assert not any("differs" in r.message for r in caplog.records)


def test_scalar_eval_value_int_spec_coerces_to_int():
    """An int-typed ScalarSpec resolves an EvalValue to int, not float."""
    s = _schema(
        {"ro_ch": ScalarSpec(label="RO ch", type=int)},
        {"ro_ch": EvalValue(expr="R1.wiring.ro_ch")},
    )
    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(kind="resonator", wiring=WiringSchema(ro_ch=2)),
            "Q1": QubitSchema(kind="qubit/transmon"),
        }
    )
    out = schema_to_raw_dict(s, parameters)["ro_ch"]
    assert out == 2
    assert isinstance(out, int) and not isinstance(out, bool)


def test_scalar_eval_value_int_spec_non_integer_raises():
    """A non-integer eval result against an int spec fails fast."""
    s = _schema(
        {"ro_ch": ScalarSpec(label="RO ch", type=int)},
        {"ro_ch": EvalValue(expr="R1.freq")},
    )
    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(kind="resonator", freq=6000.5),
            "Q1": QubitSchema(kind="qubit/transmon"),
        }
    )
    with pytest.raises(RuntimeError, match="not an integer"):
        schema_to_raw_dict(s, parameters)


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
        schema_to_raw_dict(s, None)


def test_extra_value_fields_raise():
    s = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(5), "extra": DirectValue(1)},
    )
    with pytest.raises(RuntimeError, match="extra"):
        schema_to_raw_dict(s, None)


# ---------------------------------------------------------------------------
# SweepSpec / SweepValue
# ---------------------------------------------------------------------------


def test_sweepvalue_auto_norm_derives_step_from_expts():
    # A plain SweepValue(start, stop, expts=N) self-derives step (it is a view of
    # expts, not an independent input) so any direct construction — the adapter
    # defaults, session codec, inheritance — is self-consistent. Phase 130.
    v = SweepValue(start=0.0, stop=600.0, expts=301)
    assert v.step == pytest.approx(2.0)  # (600-0)/(301-1), not the 0.1 default


def test_sweepvalue_auto_norm_false_preserves_step():
    # SweepEditor (the canonicalisation authority) passes auto_norm=False so its
    # already-computed step is not re-derived.
    v = SweepValue(start=0.0, stop=600.0, expts=301, step=0.1, auto_norm=False)
    assert v.step == pytest.approx(0.1)


def test_sweepvalue_auto_norm_expts_one_gives_zero_step():
    v = SweepValue(start=0.0, stop=1.0, expts=1)
    assert v.step == pytest.approx(0.0)


def test_sweepvalue_auto_norm_skips_evalvalue_bounds():
    # EvalValue bounds are left to SweepEditor (resolved-edge handling); auto_norm
    # never touches an EvalValue's resolved, so the supplied step stays.
    from zcu_tools.gui.cfg import EvalValue

    v = SweepValue(start=EvalValue("r_f - 0.1"), stop=600.0, expts=11, step=0.5)
    assert v.step == pytest.approx(0.5)


def test_sweep_produces_sweep_cfg():
    s = _schema(
        {"sweep": SweepSpec(label="Freq")},
        {"sweep": SweepValue(start=1.0, stop=2.0, expts=11)},
    )
    result = schema_to_raw_dict(s, None)
    sweep = result["sweep"]
    assert isinstance(sweep, dict)
    assert sweep["start"] == pytest.approx(1.0)
    assert sweep["stop"] == pytest.approx(2.0)
    assert sweep["expts"] == 11


def test_sweep_step_mode():
    s = _schema(
        {"sweep": SweepSpec()},
        {"sweep": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    result = schema_to_raw_dict(s, None)
    sweep = result["sweep"]
    assert isinstance(sweep, dict)
    assert sweep["expts"] == 11
    assert sweep["step"] == pytest.approx(0.1)


def test_sweep_eval_edges_use_resolved_value():
    s = _schema(
        {"sweep": SweepSpec()},
        {
            "sweep": SweepValue(
                start=EvalValue(expr="R1.freq - 10", resolved=5990.0),
                stop=EvalValue(expr="R1.freq + 10", resolved=6010.0),
                expts=11,
            )
        },
    )
    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(kind="resonator", freq=6000.0),
            "Q1": QubitSchema(kind="qubit/transmon"),
        }
    )
    result = schema_to_raw_dict(s, parameters)
    sweep = result["sweep"]
    assert isinstance(sweep, dict)
    assert sweep["start"] == pytest.approx(5990.0)
    assert sweep["stop"] == pytest.approx(6010.0)


def test_sweep_eval_unresolved_fails_fast():
    s = _schema(
        {"sweep": SweepSpec()},
        {
            "sweep": SweepValue(
                start=EvalValue(expr="missing", resolved=None),
                stop=1.0,
                expts=11,
            )
        },
    )
    with pytest.raises(RuntimeError) as error:
        schema_to_raw_dict(s, None)
    assert "missing" in str(error.value)


def test_sweep_eval_edges_resolve_against_md_when_no_snapshot():
    """Sweep edges built without a snapshot are resolved by lowering against md."""
    s = _schema(
        {"sweep": SweepSpec()},
        {
            "sweep": SweepValue(
                start=EvalValue(expr="R1.freq - 10"),
                stop=EvalValue(expr="R1.freq + 10"),
                expts=11,
            )
        },
    )
    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(kind="resonator", freq=6000.0),
            "Q1": QubitSchema(kind="qubit/transmon"),
        }
    )
    sweep = schema_to_raw_dict(s, parameters)["sweep"]
    assert isinstance(sweep, dict)
    assert sweep["start"] == pytest.approx(5990.0)
    assert sweep["stop"] == pytest.approx(6010.0)


# ---------------------------------------------------------------------------
# CenteredSweepSpec / CenteredSweepValue
# ---------------------------------------------------------------------------


def test_centered_sweepvalue_auto_norm_derives_step_from_span():
    v = CenteredSweepValue(center=0.0, span=100.0, expts=201)
    assert v.step == pytest.approx(0.5)


def test_centered_sweep_produces_sweep_cfg():
    s = _schema(
        {"sweep": CenteredSweepSpec(label="Freq")},
        {"sweep": CenteredSweepValue(center=10.0, span=4.0, expts=5)},
    )
    result = schema_to_raw_dict(s, None)
    sweep = result["sweep"]
    assert isinstance(sweep, dict)
    assert sweep["start"] == pytest.approx(8.0)
    assert sweep["stop"] == pytest.approx(12.0)
    assert sweep["expts"] == 5


def test_centered_sweep_single_point_lowers_to_center():
    s = _schema(
        {"sweep": CenteredSweepSpec(label="Freq")},
        {"sweep": CenteredSweepValue(center=10.0, span=4.0, expts=1)},
    )
    result = schema_to_raw_dict(s, None)
    sweep = result["sweep"]
    assert isinstance(sweep, dict)
    assert sweep["start"] == pytest.approx(10.0)
    assert sweep["stop"] == pytest.approx(10.0)
    assert sweep["expts"] == 1


def test_centered_sweep_rejects_zero_span_multi_point():
    s = _schema(
        {"sweep": CenteredSweepSpec(label="Freq")},
        {"sweep": CenteredSweepValue(center=10.0, span=0.0, expts=5)},
    )

    with pytest.raises(RuntimeError, match="span must be greater than 0"):
        schema_to_raw_dict(s, None)


def test_centered_sweep_locked_center_rejects_mismatch():
    s = _schema(
        {"sweep": CenteredSweepSpec(label="Freq", locked_center=0.0)},
        {"sweep": CenteredSweepValue(center=5.0, span=10.0, expts=11)},
    )

    with pytest.raises(RuntimeError, match="locked to 0.0"):
        schema_to_raw_dict(s, None)


def test_centered_sweep_eval_center_resolves_against_md_when_no_snapshot():
    s = _schema(
        {"sweep": CenteredSweepSpec()},
        {
            "sweep": CenteredSweepValue(
                center=EvalValue(expr="R1.freq"),
                span=4.0,
                expts=5,
            )
        },
    )
    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(kind="resonator", freq=6000.0),
            "Q1": QubitSchema(kind="qubit/transmon"),
        }
    )
    sweep = schema_to_raw_dict(s, parameters)["sweep"]
    assert isinstance(sweep, dict)
    assert sweep["start"] == pytest.approx(5998.0)
    assert sweep["stop"] == pytest.approx(6002.0)


# ---------------------------------------------------------------------------
# ReferenceSpec / ReferenceValue — value is directly flattened (no ml.get_module)
# ---------------------------------------------------------------------------


def test_module_ref_named_key_without_library_entry_raises():
    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(kind="resonator"),
            "Q1": QubitSchema(kind="qubit/transmon"),
        }
    )
    inner_spec = CfgSectionSpec(
        label="Direct Readout",
        fields={"ro_ch": ScalarSpec(label="RO ch", type=int)},
    )
    s = _schema(
        {"readout": ReferenceSpec(kind="module", allowed=[inner_spec])},
        {
            "readout": ReferenceValue(
                chosen_key="R1.readout.readout_rf",
                value=CfgSectionValue(fields={"ro_ch": DirectValue(99)}),
            )
        },
    )
    with pytest.raises(RuntimeError, match="Unknown module reference"):
        schema_to_raw_dict(s, parameters)


def test_module_ref_custom_key_resolves_by_label():
    inner_spec = CfgSectionSpec(
        label="Direct Readout",
        fields={"gain": ScalarSpec(label="Gain", type=float)},
    )
    s = _schema(
        {"ro": ReferenceSpec(kind="module", allowed=[inner_spec])},
        {
            "ro": ReferenceValue(
                chosen_key="<Custom:Direct Readout>",
                value=CfgSectionValue(fields={"gain": DirectValue(0.5)}),
            )
        },
    )
    result = schema_to_raw_dict(s, None)
    assert result["ro"] == {"gain": pytest.approx(0.5)}


# ---------------------------------------------------------------------------
# CfgSectionSpec nesting
# ---------------------------------------------------------------------------


def test_nested_section_is_recursed():
    s = _schema(
        {"inner": CfgSectionSpec(fields={"x": ScalarSpec(label="X", type=int)})},
        {"inner": CfgSectionValue(fields={"x": DirectValue(42)})},
    )
    assert schema_to_raw_dict(s, None) == {"inner": {"x": 42}}


# ---------------------------------------------------------------------------
# make_default_value
# ---------------------------------------------------------------------------


def test_make_default_value_scalar():
    spec = CfgSectionSpec(fields={"x": ScalarSpec(label="X", type=int)})
    val = make_default_value(spec)
    assert isinstance(val.fields["x"], DirectValue)


def test_make_default_value_sweep():
    spec = CfgSectionSpec(fields={"s": SweepSpec()})
    val = make_default_value(spec)
    sv = val.fields["s"]
    assert isinstance(sv, SweepValue)
    assert sv.expts == 11
    assert sv.step == pytest.approx(0.1)


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
    spec = CfgSectionSpec(
        fields={"ro": ReferenceSpec(kind="module", allowed=[inner_spec])}
    )
    val = make_default_value(spec)
    ro = val.fields["ro"]
    assert isinstance(ro, ReferenceValue)
    assert ro.chosen_key == "<Custom:Direct Readout>"
    assert isinstance(ro.value, CfgSectionValue)


# ---------------------------------------------------------------------------
# WritebackItem __post_init__ validation
# ---------------------------------------------------------------------------


def test_parameter_writeback_valid():
    item = ParameterWriteback(target_name="freq", description="d", proposed_value=1)
    assert item.target_name == "freq"
    assert item.proposed_value == 1
    # session_id is stamped by the service, not the adapter; editor identity is
    # private to the service-owned draft.
    assert item.session_id == ""


def test_module_writeback_valid():
    # target_name is the apply destination; edit_schema defaults to None and
    # editor identity is not part of the adapter item interface.
    item = ModuleWriteback(target_name="pulse_a", description="d")
    assert item.target_name == "pulse_a"
    assert item.edit_schema is None
    assert item.role_id is None
    assert not hasattr(item, "editor_id")
    assert item.session_id == ""


def test_waveform_writeback_valid():
    item = WaveformWriteback(target_name="gauss", description="d")
    assert item.target_name == "gauss"
    assert item.edit_schema is None
    assert item.role_id is None


# ---------------------------------------------------------------------------
# ReferenceSpec / ReferenceSpec empty allowed raises
# ---------------------------------------------------------------------------


def test_module_ref_spec_empty_allowed_raises():
    with pytest.raises(RuntimeError, match="allowed"):
        ReferenceSpec(kind="module", allowed=[])


def test_waveform_ref_spec_empty_allowed_raises():
    with pytest.raises(RuntimeError, match="allowed"):
        ReferenceSpec(kind="waveform", allowed=[])


# ---------------------------------------------------------------------------
# optional ReferenceSpec lowering
# ---------------------------------------------------------------------------


def _inner_module_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Pulse",
        fields={"ch": ScalarSpec(label="Ch", type=int)},
    )


def test_optional_module_ref_omitted_when_disabled():
    inner_spec = _inner_module_spec()
    outer_spec = CfgSectionSpec(
        fields={
            "module": ReferenceSpec(kind="module", allowed=[inner_spec], optional=True),
            "reps": ScalarSpec(label="Reps", type=int),
        }
    )
    # disabled optional ref = None entry (ADR-0010); the value tree is complete
    # (the key is present), and lowering omits it from the raw cfg.
    outer_val = CfgSectionValue(fields={"module": None, "reps": DirectValue(100)})
    s = CfgSchema(spec=outer_spec, value=outer_val)
    result = schema_to_raw_dict(s, None)

    assert "module" not in result
    assert result["reps"] == 100


def test_optional_module_ref_included_when_enabled():
    inner_spec = _inner_module_spec()
    outer_spec = CfgSectionSpec(
        fields={
            "module": ReferenceSpec(kind="module", allowed=[inner_spec], optional=True),
            "reps": ScalarSpec(label="Reps", type=int),
        }
    )
    inner_val = CfgSectionValue(fields={"ch": DirectValue(3)})
    outer_val = CfgSectionValue(
        fields={
            "module": ReferenceValue(chosen_key="<Custom:Pulse>", value=inner_val),
            "reps": DirectValue(100),
        }
    )
    s = CfgSchema(spec=outer_spec, value=outer_val)
    result = schema_to_raw_dict(s, None)

    assert result["reps"] == 100
    assert cast(dict, result["module"])["ch"] == 3


def test_non_optional_missing_raises():
    outer_spec = CfgSectionSpec(fields={"reps": ScalarSpec(label="Reps", type=int)})
    outer_val = CfgSectionValue(fields={})  # missing "reps"
    s = CfgSchema(spec=outer_spec, value=outer_val)

    with pytest.raises(RuntimeError, match="reps.*missing"):
        schema_to_raw_dict(s, None)


# ---------------------------------------------------------------------------
# Device selector lowering
# ---------------------------------------------------------------------------


def test_device_selector_normal_path_produces_device_name():
    spec = CfgSectionSpec(
        fields={
            "dev": ScalarSpec(
                label="Device",
                type=str,
                choices_source="devices",
                required=True,
            )
        }
    )
    val = CfgSectionValue(fields={"dev": DirectValue("lo_device")})
    s = CfgSchema(spec=spec, value=val)
    result = schema_to_raw_dict(s, None)
    assert result["dev"] == "lo_device"


def test_device_selector_required_empty_raises():
    spec = CfgSectionSpec(
        fields={
            "dev": ScalarSpec(
                label="Device",
                type=str,
                choices_source="devices",
                required=True,
            )
        }
    )
    # An empty device ref is unset → lowering raises (ADR-0010: no is_unset flag).
    val = CfgSectionValue(fields={"dev": DirectValue(value="")})
    s = CfgSchema(spec=spec, value=val)
    with pytest.raises(RuntimeError, match="required and must not be empty"):
        schema_to_raw_dict(s, None)


def test_device_selector_empty_string_raises():
    spec = CfgSectionSpec(
        fields={
            "dev": ScalarSpec(
                label="Device",
                type=str,
                choices_source="devices",
                required=True,
            )
        }
    )
    val = CfgSectionValue(fields={"dev": DirectValue(value="")})
    s = CfgSchema(spec=spec, value=val)
    with pytest.raises(RuntimeError, match="required and must not be empty"):
        schema_to_raw_dict(s, None)


# ---------------------------------------------------------------------------
# Custom reference error paths through the public lowering boundary
# ---------------------------------------------------------------------------


def test_custom_key_missing_close_bracket_raises():
    inner_spec = CfgSectionSpec(label="Pulse", fields={})
    ref_spec = ReferenceSpec(kind="module", allowed=[inner_spec])
    ref_val = ReferenceValue(chosen_key="<Custom:Pulse", value=CfgSectionValue())

    schema = CfgSchema(
        spec=CfgSectionSpec(fields={"ref": ref_spec}),
        value=CfgSectionValue(fields={"ref": ref_val}),
    )
    with pytest.raises(RuntimeError, match="Invalid custom reference key"):
        schema_to_raw_dict(schema, None)


def test_custom_key_unknown_label_raises():
    inner_spec = CfgSectionSpec(label="Pulse", fields={})
    ref_spec = ReferenceSpec(kind="module", allowed=[inner_spec])
    ref_val = ReferenceValue(chosen_key="<Custom:UnknownSpec>", value=CfgSectionValue())

    schema = CfgSchema(
        spec=CfgSectionSpec(fields={"ref": ref_spec}),
        value=CfgSectionValue(fields={"ref": ref_val}),
    )
    with pytest.raises(RuntimeError, match="Unknown custom reference label"):
        schema_to_raw_dict(schema, None)


# ---------------------------------------------------------------------------
# _resolve_sweep_edge — error paths
# ---------------------------------------------------------------------------


def test_resolve_sweep_edge_non_numeric_resolved_raises():
    parameters = parameter_capture(
        {"R1": ResonatorSchema(kind="resonator", freq=6000.0)}
    )
    spec = CfgSectionSpec(fields={"sweep": SweepSpec()})
    val = CfgSectionValue(
        fields={
            "sweep": SweepValue(
                start=EvalValue(expr="R1.freq", resolved="not_a_number", error=None),
                stop=1.0,
                expts=11,
            )
        }
    )
    s = CfgSchema(spec=spec, value=val)
    with pytest.raises(RuntimeError, match="numeric"):
        schema_to_raw_dict(s, parameters)


# ---------------------------------------------------------------------------
# BaseAdapter — make_default_save_paths error paths
# ---------------------------------------------------------------------------


def _make_concrete_adapter() -> BaseAdapter:
    """Create a minimal concrete no-analysis adapter for shared-method tests."""

    class _FakeAdapter(BaseAdapter):
        capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)
        exp_cls = MagicMock()

        @classmethod
        def cfg_definition(cls) -> MeasureCfgDefinition:
            return _empty_cfg_definition()

        def make_filename_stem(self, ctx):
            return "fake_stem"

    return _FakeAdapter()


def _make_ctx(**kwargs):
    from zcu_tools.gui.app.measure.adapter import ContextReadiness, SessionEnv

    from tests.gui.app.measure.remote._helpers import make_ctx

    defaults = dict(
        parameters=make_ctx().parameters,
        soc=None,
        soccfg=None,
        result_dir="/res",
        database_path="/db",
        active_label="label",
        readiness=ContextReadiness.ACTIVE,
    )
    defaults.update(kwargs)
    return SessionEnv(**defaults)  # type: ignore[arg-type]


def test_make_default_save_paths_raises_without_database_path():
    adapter = _make_concrete_adapter()
    ctx = _make_ctx(database_path="")
    with pytest.raises(RuntimeError, match="database_path is required"):
        adapter.make_default_save_paths(ctx)


def test_make_default_save_paths_raises_without_result_dir():
    adapter = _make_concrete_adapter()
    ctx = _make_ctx(result_dir="")
    with pytest.raises(RuntimeError, match="result_dir is required"):
        adapter.make_default_save_paths(ctx)


def test_make_default_save_paths_raises_without_active_label():
    adapter = _make_concrete_adapter()
    ctx = _make_ctx(active_label="")
    with pytest.raises(RuntimeError, match="active_label is required"):
        adapter.make_default_save_paths(ctx)


# ---------------------------------------------------------------------------
# BaseAdapter — no-analysis adapters raise (Fast Fail) on analysis methods
# ---------------------------------------------------------------------------


def test_base_adapter_get_analyze_params_returns_noparams_when_no_params():
    # A NoAnalyzeParams adapter (the 4th generic arg defaults to it) inherits the
    # base get_analyze_params, which returns the empty params instance — no
    # per-adapter boilerplate override needed.
    adapter = _make_concrete_adapter()
    from zcu_tools.gui.app.measure.adapter import SessionEnv

    assert isinstance(
        adapter.get_analyze_params(
            MagicMock(), SessionEnv(parameters=None, soc=None, soccfg=None)
        ),
        NoAnalyzeParams,
    )


def test_base_adapter_validates_forgotten_real_analyze_params_override():
    # An adapter whose analyze-params REQUIRE values (a field without a default)
    # but forgetting the override still Fast-Fails (the guard). Params whose every
    # field has a default instead inherit the base get_analyze_params (covered by
    # test_base_validation).
    from dataclasses import dataclass

    @dataclass
    class _RequiredParams:
        flag: bool  # no default → base get_analyze_params cannot construct it

    with pytest.raises(TypeError, match="get_analyze_params"):

        class _Adapter(
            BaseAdapter[MagicMock, MagicMock, NoAnalysisResult, _RequiredParams]
        ):
            capabilities = AdapterCapabilities(analysis=AnalysisMode.FIT)
            exp_cls = MagicMock()

            @classmethod
            def cfg_definition(cls) -> MeasureCfgDefinition:
                return _empty_cfg_definition()

            def make_filename_stem(self, ctx):
                return "stem"

            def analyze(self, req, *, plots: Plots):
                return NoAnalysisResult()


def test_base_adapter_analyze_raises_by_default():
    from zcu_tools.gui.app.measure.adapter import AnalyzeRequest

    adapter = _make_concrete_adapter()
    req = AnalyzeRequest(
        run_result=MagicMock(),
        analyze_params=NoAnalyzeParams(),
        parameters=None,
    )
    with pytest.raises(NotImplementedError, match="analyze"):
        adapter.analyze(req, plots=Plots(NonPresentingHost()))


# ---------------------------------------------------------------------------
# analyze_params_cls — fallback to NoAnalyzeParams when no annotation
# ---------------------------------------------------------------------------


def test_analyze_params_cls_fallback_when_no_annotation():

    class _UnannotatedAdapter(BaseAdapter):
        capabilities = AdapterCapabilities(analysis=AnalysisMode.FIT)
        exp_cls = MagicMock()

        @classmethod
        def cfg_definition(cls) -> MeasureCfgDefinition:
            return _empty_cfg_definition()

        def make_filename_stem(self, ctx):
            return "stem"

        def get_analyze_params(self, result, ctx):
            return NoAnalyzeParams()

        def analyze(self, req, *, plots: Plots):
            return NoAnalysisResult()

    # The overridden get_analyze_params has no type annotation → fallback
    cls = _UnannotatedAdapter.analyze_params_cls()
    assert cls is NoAnalyzeParams
