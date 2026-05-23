"""Phase 9.5 tests — FakeFreqAdapter simulate + analyze flow."""

from __future__ import annotations

import numpy as np
from zcu_tools.experiment.v2_gui.adapters.onetone.fakefreq import (
    FakeFreqAdapter,
    FakeFreqAnalyzeResult,
    FreqRunResult,
)
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.adapter import (
    AnalyzeRequest,
    CfgSchema,
    DirectValue,
    ExpContext,
    RunRequest,
)
from zcu_tools.gui.registry import Registry
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        res_name="unknown_resonator",
        result_dir="/tmp/zcu_result",
        database_path="/tmp/zcu_db/unknown_chip/unknown_qubit",
        active_label="ctx001",
    )


def _adapter() -> FakeFreqAdapter:
    return FakeFreqAdapter(fast_mode=True)


def _run(adapter: FakeFreqAdapter, ctx: ExpContext) -> FreqRunResult:
    schema = adapter.make_default_cfg(ctx)
    return adapter.run(_run_req(ctx), schema)


def _run_req(ctx: ExpContext) -> RunRequest:
    return RunRequest(md=ctx.md, ml=ctx.ml, soc=ctx.soc, soccfg=ctx.soccfg)


def _analyze_req(
    ctx: ExpContext,
    result: FreqRunResult,
    analyze_params: dict[str, object],
) -> AnalyzeRequest:
    return AnalyzeRequest(
        run_result=result,
        analyze_params=analyze_params,
        md=ctx.md,
        ml=ctx.ml,
        predictor=ctx.predictor,
    )


def _default_analyze_params(
    adapter: FakeFreqAdapter, result: FreqRunResult, ctx: ExpContext
) -> dict[str, object]:
    params = adapter.get_analyze_params(result, ctx)
    return {param.key: param.default for param in params}


# ---------------------------------------------------------------------------
# make_default_cfg
# ---------------------------------------------------------------------------


def test_make_default_cfg_returns_cfg_schema():
    adapter = _adapter()
    schema = adapter.make_default_cfg(_make_ctx())
    assert isinstance(schema, CfgSchema)


def test_make_default_cfg_has_expected_fields():
    adapter = _adapter()
    schema = adapter.make_default_cfg(_make_ctx())
    fields = schema.spec.fields
    for key in (
        "reps",
        "rounds",
        "freq",
        "res_freq",
        "Ql",
    ):
        assert key in fields, f"missing spec field: {key}"


def test_make_default_cfg_uses_r_f_from_md():
    """make_default_cfg should pre-fill res_freq and sweep range from md.r_f."""
    from zcu_tools.gui.adapter import SweepValue

    ctx = _make_ctx()
    ctx.md.r_f = 7500.0
    schema = _adapter().make_default_cfg(ctx)
    val = schema.value
    res_freq_val = val.fields["res_freq"]
    assert isinstance(res_freq_val, DirectValue)
    assert abs(res_freq_val.value - 7500.0) < 1e-6
    sweep = val.fields["freq"]
    assert isinstance(sweep, SweepValue)
    assert abs(sweep.start - 7300.0) < 1e-6  # r_f - 200 (no rf_w)
    assert abs(sweep.stop - 7700.0) < 1e-6


def test_make_default_cfg_uses_rf_w_for_sweep_and_ql():
    """make_default_cfg should estimate Ql and sweep range from md.r_f + md.rf_w."""
    from zcu_tools.gui.adapter import SweepValue

    ctx = _make_ctx()
    ctx.md.r_f = 6000.0
    ctx.md.rf_w = 2.0  # 2 MHz linewidth → Ql ≈ 3000, span ±10 MHz
    schema = _adapter().make_default_cfg(ctx)
    val = schema.value
    sweep = val.fields["freq"]
    assert isinstance(sweep, SweepValue)
    assert abs(sweep.start - 5990.0) < 1e-6
    assert abs(sweep.stop - 6010.0) < 1e-6
    ql_val = val.fields["Ql"]
    assert isinstance(ql_val, DirectValue)
    assert abs(ql_val.value - 3000) < 10  # round(6000/2) = 3000


def test_make_default_cfg_prefers_named_readout_module_from_ml():
    from zcu_tools.gui.adapter import CfgSectionValue
    from zcu_tools.program.v2 import ModuleCfgFactory

    ctx = _make_ctx()
    ctx.ml.register_module(
        readout_rf=ModuleCfgFactory.from_raw(
            {
                "type": "readout/direct",
                "ro_ch": 3,
                "ro_freq": 6123.0,
                "ro_length": 1.2,
                "trig_offset": 0.4,
            },
            ml=ctx.ml,
        )
    )

    schema = _adapter().make_default_cfg(ctx)
    modules = schema.value.fields["modules"]

    assert isinstance(modules, CfgSectionValue)
    readout = modules.fields["readout"]
    assert readout.chosen_key == "readout_rf"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# run — returns FreqRunResult dataclass
# ---------------------------------------------------------------------------


def test_run_returns_freq_run_result():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    assert isinstance(result, FreqRunResult)


def test_run_returns_tuple_of_arrays():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    assert isinstance(result.freqs, np.ndarray)
    assert isinstance(result.signals, np.ndarray)


def test_run_shapes_match():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    assert result.freqs.shape == result.signals.shape


def test_run_freq_range_matches_cfg():
    ctx = _make_ctx()
    adapter = _adapter()
    schema = adapter.make_default_cfg(ctx)
    # default SweepValue: start=5800.0, stop=6200.0, expts=201
    result = adapter.run(_run_req(ctx), schema)
    assert abs(result.freqs[0] - 5800.0) < 1.0
    assert abs(result.freqs[-1] - 6200.0) < 1.0
    assert len(result.freqs) == 201


def test_run_signals_are_complex():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    assert np.iscomplexobj(result.signals)


def test_run_cfg_snapshot_stored():
    ctx = _make_ctx()
    adapter = FakeFreqAdapter(fast_mode=True)
    result = _run(adapter, ctx)
    assert result.cfg_snapshot is not None
    assert result.cfg_snapshot.fast_mode is True


def test_run_noise_decreases_with_more_rounds():
    """More rounds → lower noise (SNR ∝ sqrt(rounds)).

    Run with a large noise_scale so the noise variance dominates the signal.
    Compare std of raw amplitudes between 1 round and 100 rounds.  At 100
    rounds the theoretical reduction is 10×; asserting > 3× gives a robust
    lower bound even with an unlucky random seed.
    """
    ctx = _make_ctx()
    adapter = FakeFreqAdapter(fast_mode=True)

    def _schema_with(rounds: int) -> CfgSchema:
        s = adapter.make_default_cfg(ctx)
        s.value.fields["rounds"] = DirectValue(rounds)
        s.value.fields["noise_scale"] = DirectValue(10.0)
        return s

    res_low = adapter.run(_run_req(ctx), _schema_with(1))
    res_high = adapter.run(_run_req(ctx), _schema_with(100))

    noise_low = float(np.std(np.abs(res_low.signals)))
    noise_high = float(np.std(np.abs(res_high.signals)))
    assert noise_low > noise_high * 3  # theory ≈ 10×; 3× is a safe lower bound


# ---------------------------------------------------------------------------
# analyze — returns FakeFreqAnalyzeResult dataclass
# ---------------------------------------------------------------------------


def test_analyze_returns_fake_freq_analyze_result():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    analyze_result = _adapter().analyze(
        _analyze_req(ctx, result, _default_analyze_params(_adapter(), result, ctx))
    )
    assert isinstance(analyze_result, FakeFreqAnalyzeResult)


def test_get_analyze_params_returns_flat_param_list():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    params = _adapter().get_analyze_params(result, ctx)
    assert [param.key for param in params] == ["model_type", "fit_bg_slope"]


def test_analyze_has_expected_fields():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    ar = _adapter().analyze(
        _analyze_req(ctx, result, _default_analyze_params(_adapter(), result, ctx))
    )
    assert isinstance(ar.freq, float)
    assert isinstance(ar.fwhm, float)
    assert isinstance(ar.params, dict)
    assert ar.run_result is result


def test_analyze_freq_close_to_true_value():
    """Fitted frequency should be within 5 MHz of the ground-truth 6000.0 MHz."""
    ctx = _make_ctx()
    adapter = FakeFreqAdapter(fast_mode=True)
    schema = adapter.make_default_cfg(ctx)
    schema.value.fields["rounds"] = DirectValue(200)
    result = adapter.run(_run_req(ctx), schema)
    ar = adapter.analyze(
        _analyze_req(ctx, result, _default_analyze_params(adapter, result, ctx))
    )
    assert abs(ar.freq - 6000.0) < 5.0


def test_analyze_produces_figure():
    from matplotlib.figure import Figure

    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    ar = _adapter().analyze(
        _analyze_req(ctx, result, _default_analyze_params(_adapter(), result, ctx))
    )
    assert isinstance(ar.figure, Figure)


def test_analyze_result_carries_figure():
    from matplotlib.figure import Figure

    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    ar = _adapter().analyze(
        _analyze_req(ctx, result, _default_analyze_params(_adapter(), result, ctx))
    )
    assert isinstance(ar.figure, Figure)


# ---------------------------------------------------------------------------
# writeback items
# ---------------------------------------------------------------------------


def test_get_writeback_items_has_r_f_and_rf_w():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    ar = _adapter().analyze(
        _analyze_req(ctx, result, _default_analyze_params(_adapter(), result, ctx))
    )
    spec = _adapter().get_writeback_items(ar, ctx)
    keys = [item.key for item in spec]
    assert "r_f" in keys
    assert "rf_w" in keys


# ---------------------------------------------------------------------------
# save paths / registry
# ---------------------------------------------------------------------------


def test_make_save_paths_returns_save_paths():
    from zcu_tools.gui.adapter import SavePaths

    ctx = _make_ctx()
    paths = _adapter().make_save_paths(ctx)
    assert isinstance(paths, SavePaths)
    assert paths.data_path
    assert paths.image_path
    assert "unknown_resonator_freq_" in paths.data_path
    assert "/exps/ctx001/image/" in paths.image_path


def test_registered_in_registry():
    registry = Registry()
    register_all(registry)
    assert registry.has("onetone/fake_freq")
    adapter = registry.create("onetone/fake_freq")
    assert isinstance(adapter, FakeFreqAdapter)


def test_make_default_cfg_has_mod_ref_field():
    from zcu_tools.gui.adapter import ModuleRefSpec

    ctx = _make_ctx()
    adapter = _adapter()
    schema = adapter.make_default_cfg(ctx)
    modules_spec = schema.spec.fields["modules"]
    assert hasattr(modules_spec, "fields"), "modules should be a CfgSectionSpec"
    readout_spec = modules_spec.fields["readout"]  # type: ignore[union-attr]
    assert isinstance(readout_spec, ModuleRefSpec)


def test_writeback_items_include_ml_targets():
    ctx = _make_ctx()
    adapter = _adapter()
    result = _run(adapter, ctx)
    ar = adapter.analyze(
        _analyze_req(ctx, result, _default_analyze_params(adapter, result, ctx))
    )
    spec = adapter.get_writeback_items(ar, ctx)
    keys = [item.key for item in spec]
    assert "readout_rf" in keys
    assert "ro_waveform" in keys


def test_writeback_items_include_ml_targets_when_missing():
    ctx = _make_ctx()
    adapter = _adapter()
    result = _run(adapter, ctx)
    ar = adapter.analyze(
        _analyze_req(ctx, result, _default_analyze_params(adapter, result, ctx))
    )
    spec = adapter.get_writeback_items(ar, ctx)
    keys = [item.key for item in spec]
    assert "readout_rf" in keys
    assert "ro_waveform" in keys


def test_writeback_edit_schema_provided():
    """get_writeback_items should provide edit_schema for readout_rf and ro_waveform."""
    from zcu_tools.gui.adapter import ModuleWriteback, WaveformWriteback

    ctx = _make_ctx()
    adapter = _adapter()
    result = _run(adapter, ctx)
    ar = adapter.analyze(
        _analyze_req(ctx, result, _default_analyze_params(adapter, result, ctx))
    )
    spec = adapter.get_writeback_items(ar, ctx)

    spec_by_key = {item.key: item for item in spec}
    readout_item = spec_by_key["readout_rf"]
    waveform_item = spec_by_key["ro_waveform"]
    assert isinstance(readout_item, ModuleWriteback)
    assert isinstance(waveform_item, WaveformWriteback)
    assert readout_item.edit_schema is not None
    assert waveform_item.edit_schema is not None
    assert isinstance(readout_item.edit_schema, CfgSchema)
    assert isinstance(waveform_item.edit_schema, CfgSchema)


def test_no_last_cfg_side_channel():
    """FakeFreqAdapter must not set last_cfg as an instance attribute after run."""
    ctx = _make_ctx()
    adapter = _adapter()
    _run(adapter, ctx)
    assert "last_cfg" not in adapter.__dict__
