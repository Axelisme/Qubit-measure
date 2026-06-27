"""Unit tests for zcu_tools.gui.app.main.adapter (Spec/Value split)."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    MetaDictWriteback,
    ModuleRefSpec,
    ModuleRefValue,
    ModuleWriteback,
    NoAnalysisResult,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformWriteback,
    make_default_value,
    require_soc_handles,
)
from zcu_tools.gui.app.main.adapter.lowering import find_allowed_spec
from zcu_tools.gui.app.main.adapter.protocol import NoAnalyzeParams
from zcu_tools.meta_tool import MetaDict

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


def test_require_soc_handles_is_framework_request_validation() -> None:
    from zcu_tools.gui.app.main.adapter import RunRequest

    with pytest.raises(RuntimeError, match="soc is required"):
        require_soc_handles(
            RunRequest(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=MagicMock())
        )


# ---------------------------------------------------------------------------
# ScalarSpec / ScalarValue
# ---------------------------------------------------------------------------


def test_scalar_int():
    s = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(100)},
    )
    assert s.to_raw_dict(None, _make_ml()) == {"reps": 100}


def test_scalar_str():
    s = _schema(
        {"name": ScalarSpec(label="Name", type=str)},
        {"name": DirectValue("hello")},
    )
    assert s.to_raw_dict(None, _make_ml())["name"] == "hello"


def test_scalar_editable_false_still_included():
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float, editable=False)},
        {"freq": DirectValue(6.0)},
    )
    assert s.to_raw_dict(None, _make_ml())["freq"] == pytest.approx(6.0)


def test_scalar_eval_value_uses_resolved_snapshot():
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="r_f - rf_w", resolved=5998.0)},
    )
    assert s.to_raw_dict(None, _make_ml())["freq"] == pytest.approx(5998.0)


def test_scalar_eval_value_unresolved_raises():
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="missing_name", resolved=None)},
    )
    with pytest.raises(RuntimeError, match="freq.*missing_name.*unresolved"):
        s.to_raw_dict(None, _make_ml())


def test_scalar_eval_value_resolves_against_md_when_no_snapshot():
    """An EvalValue built without a snapshot is resolved by lowering against md."""
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="r_f - rf_w")},
    )
    md = MetaDict()
    md.r_f = 6000.0
    md.rf_w = 2.0
    assert s.to_raw_dict(md, _make_ml())["freq"] == pytest.approx(5998.0)


def test_scalar_eval_snapshot_drift_warns_but_keeps_snapshot(caplog):
    """When md is supplied and the snapshot disagrees with a fresh evaluation,
    lowering keeps the snapshot but logs a drift warning (commit cross-check)."""
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="r_f - rf_w", resolved=5998.0)},
    )
    md = MetaDict()
    md.r_f = 7000.0  # md changed after the snapshot was taken
    md.rf_w = 2.0
    with caplog.at_level("WARNING"):
        out = s.to_raw_dict(md, _make_ml())
    # snapshot still wins
    assert out["freq"] == pytest.approx(5998.0)
    assert any(
        "differs from current md evaluation" in r.message for r in caplog.records
    )


def test_scalar_eval_snapshot_consistent_no_warn(caplog):
    s = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue(expr="r_f - rf_w", resolved=5998.0)},
    )
    md = MetaDict()
    md.r_f = 6000.0
    md.rf_w = 2.0  # fresh eval == snapshot
    with caplog.at_level("WARNING"):
        s.to_raw_dict(md, _make_ml())
    assert not any("differs" in r.message for r in caplog.records)


def test_scalar_eval_value_int_spec_coerces_to_int():
    """An int-typed ScalarSpec resolves an EvalValue to int, not float."""
    s = _schema(
        {"ro_ch": ScalarSpec(label="RO ch", type=int)},
        {"ro_ch": EvalValue(expr="ro_ch")},
    )
    md = MetaDict()
    md.ro_ch = 2
    out = s.to_raw_dict(md, _make_ml())["ro_ch"]
    assert out == 2
    assert isinstance(out, int) and not isinstance(out, bool)


def test_scalar_eval_value_int_spec_non_integer_raises():
    """A non-integer eval result against an int spec fails fast."""
    s = _schema(
        {"ro_ch": ScalarSpec(label="RO ch", type=int)},
        {"ro_ch": EvalValue(expr="r_f")},
    )
    md = MetaDict()
    md.r_f = 6000.5
    with pytest.raises(RuntimeError, match="not an integer"):
        s.to_raw_dict(md, _make_ml())


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
        s.to_raw_dict(None, _make_ml())


def test_extra_value_fields_raise():
    s = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(5), "extra": DirectValue(1)},
    )
    with pytest.raises(RuntimeError, match="extra"):
        s.to_raw_dict(None, _make_ml())


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
    from zcu_tools.gui.app.main.adapter import EvalValue

    v = SweepValue(start=EvalValue("r_f - 0.1"), stop=600.0, expts=11, step=0.5)
    assert v.step == pytest.approx(0.5)


def test_sweep_produces_sweep_cfg():
    from zcu_tools.program.v2 import SweepCfg

    s = _schema(
        {"sweep": SweepSpec(label="Freq")},
        {"sweep": SweepValue(start=1.0, stop=2.0, expts=11)},
    )
    result = s.to_raw_dict(None, _make_ml())
    sweep = result["sweep"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.start == pytest.approx(1.0)
    assert sweep.stop == pytest.approx(2.0)
    assert sweep.expts == 11


def test_sweep_step_mode():
    from zcu_tools.program.v2 import SweepCfg

    s = _schema(
        {"sweep": SweepSpec()},
        {"sweep": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    result = s.to_raw_dict(None, _make_ml())
    sweep = result["sweep"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.expts == 11
    assert sweep.step == pytest.approx(0.1)


def test_sweep_eval_edges_use_resolved_value():
    from zcu_tools.program.v2 import SweepCfg

    s = _schema(
        {"sweep": SweepSpec()},
        {
            "sweep": SweepValue(
                start=EvalValue(expr="r_f - 10", resolved=5990.0),
                stop=EvalValue(expr="r_f + 10", resolved=6010.0),
                expts=11,
            )
        },
    )
    result = s.to_raw_dict(None, _make_ml())
    sweep = result["sweep"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.start == pytest.approx(5990.0)
    assert sweep.stop == pytest.approx(6010.0)


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
    with pytest.raises(RuntimeError, match="unresolved"):
        s.to_raw_dict(None, _make_ml())


def test_sweep_eval_edges_resolve_against_md_when_no_snapshot():
    """Sweep edges built without a snapshot are resolved by lowering against md."""
    from zcu_tools.program.v2 import SweepCfg

    s = _schema(
        {"sweep": SweepSpec()},
        {
            "sweep": SweepValue(
                start=EvalValue(expr="r_f - 10"),
                stop=EvalValue(expr="r_f + 10"),
                expts=11,
            )
        },
    )
    md = MetaDict()
    md.r_f = 6000.0
    sweep = s.to_raw_dict(md, _make_ml())["sweep"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.start == pytest.approx(5990.0)
    assert sweep.stop == pytest.approx(6010.0)


# ---------------------------------------------------------------------------
# ModuleRefSpec / ModuleRefValue — value is directly flattened (no ml.get_module)
# ---------------------------------------------------------------------------


def test_module_ref_named_key_without_library_entry_raises():
    ml = _make_ml()
    ml.modules = {}
    ml.waveforms = {}
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
    with pytest.raises(RuntimeError, match="Unknown module reference"):
        s.to_raw_dict(None, ml)


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
    result = s.to_raw_dict(None, _make_ml())
    assert result["ro"] == {"gain": pytest.approx(0.5)}


# ---------------------------------------------------------------------------
# CfgSectionSpec nesting
# ---------------------------------------------------------------------------


def test_nested_section_is_recursed():
    s = _schema(
        {"inner": CfgSectionSpec(fields={"x": ScalarSpec(label="X", type=int)})},
        {"inner": CfgSectionValue(fields={"x": DirectValue(42)})},
    )
    assert s.to_raw_dict(None, _make_ml()) == {"inner": {"x": 42}}


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
    spec = CfgSectionSpec(fields={"ro": ModuleRefSpec(allowed=[inner_spec])})
    val = make_default_value(spec)
    ro = val.fields["ro"]
    assert isinstance(ro, ModuleRefValue)
    assert ro.chosen_key == "<Custom:Direct Readout>"
    assert isinstance(ro.value, CfgSectionValue)


# ---------------------------------------------------------------------------
# WritebackItem __post_init__ validation
# ---------------------------------------------------------------------------


def test_meta_dict_writeback_requires_proposed_value():
    with pytest.raises(TypeError):
        MetaDictWriteback(target_name="k", description="d")  # type: ignore[call-arg]


def test_meta_dict_writeback_valid():
    item = MetaDictWriteback(target_name="freq", description="d", proposed_value=1)
    assert item.target_name == "freq"
    assert item.proposed_value == 1
    # session_id / editor_id are stamped by the service, not the adapter.
    assert item.session_id == ""


def test_module_writeback_valid():
    # target_name is the apply destination; edit_schema/editor_id default to None.
    item = ModuleWriteback(target_name="pulse_a", description="d")
    assert item.target_name == "pulse_a"
    assert item.edit_schema is None
    assert item.editor_id is None
    assert item.session_id == ""


def test_waveform_writeback_valid():
    item = WaveformWriteback(target_name="gauss", description="d")
    assert item.target_name == "gauss"
    assert item.edit_schema is None


# ---------------------------------------------------------------------------
# ModuleRefSpec / WaveformRefSpec empty allowed raises
# ---------------------------------------------------------------------------


def test_module_ref_spec_empty_allowed_raises():
    with pytest.raises(RuntimeError, match="allowed"):
        ModuleRefSpec(allowed=[])


def test_waveform_ref_spec_empty_allowed_raises():
    with pytest.raises(RuntimeError, match="allowed"):
        WaveformRefSpec(allowed=[])


# ---------------------------------------------------------------------------
# optional ModuleRefSpec lowering
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
            "module": ModuleRefSpec(allowed=[inner_spec], optional=True),
            "reps": ScalarSpec(label="Reps", type=int),
        }
    )
    # disabled optional ref = None entry (ADR-0010); the value tree is complete
    # (the key is present), and lowering omits it from the raw cfg.
    outer_val = CfgSectionValue(fields={"module": None, "reps": DirectValue(100)})
    s = CfgSchema(spec=outer_spec, value=outer_val)
    result = s.to_raw_dict(None, _make_ml())

    assert "module" not in result
    assert result["reps"] == 100


def test_optional_module_ref_included_when_enabled():
    inner_spec = _inner_module_spec()
    outer_spec = CfgSectionSpec(
        fields={
            "module": ModuleRefSpec(allowed=[inner_spec], optional=True),
            "reps": ScalarSpec(label="Reps", type=int),
        }
    )
    inner_val = CfgSectionValue(fields={"ch": DirectValue(3)})
    outer_val = CfgSectionValue(
        fields={
            "module": ModuleRefValue(chosen_key="<Custom:Pulse>", value=inner_val),
            "reps": DirectValue(100),
        }
    )
    s = CfgSchema(spec=outer_spec, value=outer_val)
    result = s.to_raw_dict(None, _make_ml())

    assert result["reps"] == 100
    assert cast(dict, result["module"])["ch"] == 3


def test_non_optional_missing_raises():
    outer_spec = CfgSectionSpec(fields={"reps": ScalarSpec(label="Reps", type=int)})
    outer_val = CfgSectionValue(fields={})  # missing "reps"
    s = CfgSchema(spec=outer_spec, value=outer_val)

    with pytest.raises(RuntimeError, match="reps.*missing"):
        s.to_raw_dict(None, _make_ml())


# ---------------------------------------------------------------------------
# DeviceRefSpec lowering
# ---------------------------------------------------------------------------


def test_device_ref_normal_path_produces_device_name():
    spec = CfgSectionSpec(fields={"dev": DeviceRefSpec(label="Device")})
    val = CfgSectionValue(fields={"dev": DirectValue("lo_device")})
    s = CfgSchema(spec=spec, value=val)
    result = s.to_raw_dict(None, _make_ml())
    assert result["dev"] == "lo_device"


def test_device_ref_is_unset_raises():
    spec = CfgSectionSpec(fields={"dev": DeviceRefSpec(label="Device")})
    # An empty device ref is unset → lowering raises (ADR-0010: no is_unset flag).
    val = CfgSectionValue(fields={"dev": DirectValue(value="")})
    s = CfgSchema(spec=spec, value=val)
    with pytest.raises(RuntimeError, match="unset"):
        s.to_raw_dict(None, _make_ml())


def test_device_ref_empty_string_raises():
    spec = CfgSectionSpec(fields={"dev": DeviceRefSpec(label="Device")})
    val = CfgSectionValue(fields={"dev": DirectValue(value="")})
    s = CfgSchema(spec=spec, value=val)
    with pytest.raises(RuntimeError, match="unset"):
        s.to_raw_dict(None, _make_ml())


# ---------------------------------------------------------------------------
# find_allowed_spec — Custom key error paths
# ---------------------------------------------------------------------------


def testfind_allowed_spec_custom_key_missing_close_bracket_raises():
    inner_spec = CfgSectionSpec(label="Pulse", fields={})
    ref_spec = ModuleRefSpec(allowed=[inner_spec])
    ref_val = ModuleRefValue(chosen_key="<Custom:Pulse", value=CfgSectionValue())

    with pytest.raises(RuntimeError, match="Invalid custom reference key"):
        find_allowed_spec(ref_spec, ref_val, None)


def testfind_allowed_spec_custom_key_unknown_label_raises():
    inner_spec = CfgSectionSpec(label="Pulse", fields={})
    ref_spec = ModuleRefSpec(allowed=[inner_spec])
    ref_val = ModuleRefValue(chosen_key="<Custom:UnknownSpec>", value=CfgSectionValue())

    with pytest.raises(RuntimeError, match="Unknown custom reference label"):
        find_allowed_spec(ref_spec, ref_val, None)


# ---------------------------------------------------------------------------
# _resolve_sweep_edge — error paths
# ---------------------------------------------------------------------------


def test_resolve_sweep_edge_unresolved_eval_raises():
    spec = CfgSectionSpec(fields={"sweep": SweepSpec()})
    val = CfgSectionValue(
        fields={
            "sweep": SweepValue(
                start=EvalValue(expr="r_f", resolved=None, error=None),
                stop=1.0,
                expts=11,
            )
        }
    )
    s = CfgSchema(spec=spec, value=val)
    with pytest.raises(RuntimeError, match="unresolved"):
        s.to_raw_dict(None, None)


def test_resolve_sweep_edge_non_numeric_resolved_raises():
    spec = CfgSectionSpec(fields={"sweep": SweepSpec()})
    val = CfgSectionValue(
        fields={
            "sweep": SweepValue(
                start=EvalValue(expr="r_f", resolved="not_a_number", error=None),
                stop=1.0,
                expts=11,
            )
        }
    )
    s = CfgSchema(spec=spec, value=val)
    with pytest.raises(RuntimeError, match="non-numeric"):
        s.to_raw_dict(None, None)


# ---------------------------------------------------------------------------
# BaseAdapter — make_default_save_paths error paths
# ---------------------------------------------------------------------------


def _make_concrete_adapter() -> BaseAdapter:
    """Create a minimal concrete no-analysis adapter for shared-method tests."""

    class _FakeAdapter(BaseAdapter):
        capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)
        exp_cls = MagicMock()

        @classmethod
        def cfg_spec(cls):
            return CfgSectionSpec()

        def make_default_value(self, ctx):
            return CfgSectionValue()

        def build_exp_cfg(self, raw_cfg, req):
            return MagicMock()

        def make_filename_stem(self, ctx):
            return "fake_stem"

    return _FakeAdapter()


def _make_ctx(**kwargs):
    from zcu_tools.gui.app.main.adapter import ContextReadiness, ExpContext
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    defaults = dict(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        result_dir="/res",
        database_path="/db",
        active_label="label",
        readiness=ContextReadiness.ACTIVE,
    )
    defaults.update(kwargs)
    return ExpContext(**defaults)  # type: ignore[arg-type]


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
    assert isinstance(
        adapter.get_analyze_params(MagicMock(), _make_ctx()), NoAnalyzeParams
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
            def cfg_spec(cls):
                return CfgSectionSpec()

            def make_default_value(self, ctx):
                return CfgSectionValue()

            def build_exp_cfg(self, raw_cfg, req):
                return MagicMock()

            def make_filename_stem(self, ctx):
                return "stem"

            def analyze(self, req):
                return NoAnalysisResult()


def test_base_adapter_analyze_raises_by_default():
    from zcu_tools.gui.app.main.adapter import AnalyzeRequest

    adapter = _make_concrete_adapter()
    req = AnalyzeRequest(
        run_result=MagicMock(),
        analyze_params=NoAnalyzeParams(),
        md=MagicMock(),
        ml=MagicMock(),
        predictor=None,
    )
    with pytest.raises(NotImplementedError, match="analyze"):
        adapter.analyze(req)


# ---------------------------------------------------------------------------
# BaseAdapter — default build_exp_cfg via ExpCfg_cls
# ---------------------------------------------------------------------------


def test_base_adapter_build_exp_cfg_delegates_to_make_cfg():
    """ExpCfg_cls set → default build_exp_cfg delegates to cfg_assembler.make_cfg."""

    class _Cfg:
        pass

    class _Adapter(BaseAdapter):
        capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)
        exp_cls = MagicMock()
        ExpCfg_cls = _Cfg

        @classmethod
        def cfg_spec(cls):
            return CfgSectionSpec()

        def make_default_value(self, ctx):
            return CfgSectionValue()

        def make_filename_stem(self, ctx):
            return "stem"

    req = MagicMock()
    sentinel = object()
    with patch(
        "zcu_tools.experiment.v2_gui.adapters.base.make_cfg", return_value=sentinel
    ) as make_cfg:
        out = _Adapter().build_exp_cfg({"reps": 1}, req)
    make_cfg.assert_called_once_with({"reps": 1}, _Cfg, ml=req.ml)
    assert out is sentinel


def test_base_adapter_build_exp_cfg_raises_without_expcfg_cls():
    """Neither ExpCfg_cls set nor build_exp_cfg overridden → Fast Fail."""

    class _Adapter(BaseAdapter):
        capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)
        exp_cls = MagicMock()

        @classmethod
        def cfg_spec(cls):
            return CfgSectionSpec()

        def make_default_value(self, ctx):
            return CfgSectionValue()

        def make_filename_stem(self, ctx):
            return "stem"

    with pytest.raises(NotImplementedError, match="ExpCfg_cls"):
        _Adapter().build_exp_cfg({}, MagicMock())


# ---------------------------------------------------------------------------
# analyze_params_cls — fallback to NoAnalyzeParams when no annotation
# ---------------------------------------------------------------------------


def test_analyze_params_cls_fallback_when_no_annotation():

    class _UnannotatedAdapter(BaseAdapter):
        capabilities = AdapterCapabilities(analysis=AnalysisMode.FIT)
        exp_cls = MagicMock()

        @classmethod
        def cfg_spec(cls):
            return CfgSectionSpec()

        def make_default_value(self, ctx):
            return CfgSectionValue()

        def build_exp_cfg(self, raw_cfg, req):
            return MagicMock()

        def make_filename_stem(self, ctx):
            return "stem"

        def get_analyze_params(self, result, ctx):
            return NoAnalyzeParams()

        def analyze(self, req):
            return NoAnalysisResult()

    # The overridden get_analyze_params has no type annotation → fallback
    cls = _UnannotatedAdapter.analyze_params_cls()
    assert cls is NoAnalyzeParams
