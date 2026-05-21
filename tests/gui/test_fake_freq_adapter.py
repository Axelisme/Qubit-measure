"""Phase 9.5 tests — FakeFreqAdapter simulate + analyze flow."""

from __future__ import annotations

import numpy as np
from zcu_tools.experiment.v2_gui.adapters.onetone.freq import (
    FakeFreqAdapter,
    FakeFreqAnalyzeResult,
    FreqRunResult,
)
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.adapter import CfgSchema, ExpContext
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
    )


def _adapter() -> FakeFreqAdapter:
    return FakeFreqAdapter(fast_mode=True)


def _run(adapter: FakeFreqAdapter, ctx: ExpContext) -> FreqRunResult:
    schema = adapter.make_default_cfg(ctx)
    return adapter.run(ctx, schema)


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
    from zcu_tools.gui.adapter import ScalarValue, SweepValue

    ctx = _make_ctx()
    ctx.md.r_f = 7500.0
    schema = _adapter().make_default_cfg(ctx)
    val = schema.value
    res_freq_val = val.fields["res_freq"]
    assert isinstance(res_freq_val, ScalarValue)
    assert abs(res_freq_val.value - 7500.0) < 1e-6
    sweep = val.fields["freq"]
    assert isinstance(sweep, SweepValue)
    assert abs(sweep.start - 7300.0) < 1e-6  # r_f - 200 (no rf_w)
    assert abs(sweep.stop - 7700.0) < 1e-6


def test_make_default_cfg_uses_rf_w_for_sweep_and_ql():
    """make_default_cfg should estimate Ql and sweep range from md.r_f + md.rf_w."""
    from zcu_tools.gui.adapter import ScalarValue, SweepValue

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
    assert isinstance(ql_val, ScalarValue)
    assert abs(ql_val.value - 3000) < 10  # round(6000/2) = 3000


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
    result = adapter.run(ctx, schema)
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
    from zcu_tools.gui.adapter import ScalarValue

    ctx = _make_ctx()
    adapter = FakeFreqAdapter(fast_mode=True)

    def _schema_with(rounds: int) -> CfgSchema:
        s = adapter.make_default_cfg(ctx)
        s.value.fields["rounds"] = ScalarValue(rounds)
        s.value.fields["noise_scale"] = ScalarValue(10.0)
        return s

    res_low = adapter.run(ctx, _schema_with(1))
    res_high = adapter.run(ctx, _schema_with(100))

    noise_low = float(np.std(np.abs(res_low.signals)))
    noise_high = float(np.std(np.abs(res_high.signals)))
    assert noise_low > noise_high * 3  # theory ≈ 10×; 3× is a safe lower bound


# ---------------------------------------------------------------------------
# analyze — returns FakeFreqAnalyzeResult dataclass
# ---------------------------------------------------------------------------


def test_analyze_returns_fake_freq_analyze_result():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    analyze_result = _adapter().analyze(result, ctx)
    assert isinstance(analyze_result, FakeFreqAnalyzeResult)


def test_analyze_has_expected_fields():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    ar = _adapter().analyze(result, ctx)
    assert isinstance(ar.freq, float)
    assert isinstance(ar.fwhm, float)
    assert isinstance(ar.params, dict)
    assert ar.run_result is result


def test_analyze_freq_close_to_true_value():
    """Fitted frequency should be within 5 MHz of the ground-truth 6000.0 MHz."""
    from zcu_tools.gui.adapter import ScalarValue

    ctx = _make_ctx()
    adapter = FakeFreqAdapter(fast_mode=True)
    schema = adapter.make_default_cfg(ctx)
    schema.value.fields["rounds"] = ScalarValue(200)
    result = adapter.run(ctx, schema)
    ar = adapter.analyze(result, ctx)
    assert abs(ar.freq - 6000.0) < 5.0


def test_analyze_produces_figure():
    from matplotlib.figure import Figure

    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    ar = _adapter().analyze(result, ctx)
    assert isinstance(ar.figure, Figure)


def test_get_figure_returns_figure():
    from matplotlib.figure import Figure

    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    ar = _adapter().analyze(result, ctx)
    fig = _adapter().get_figure(ar)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# writeback spec
# ---------------------------------------------------------------------------


def test_get_writeback_spec_has_r_f_and_rf_w():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    ar = _adapter().analyze(result, ctx)
    spec = _adapter().get_writeback_spec(ar, ctx)
    keys = [item.key for item in spec]
    assert "r_f" in keys
    assert "rf_w" in keys


def test_apply_writeback_sets_md_r_f():
    ctx = _make_ctx()
    result = _run(_adapter(), ctx)
    ar = _adapter().analyze(result, ctx)
    _adapter().apply_writeback(ctx, ar, ["r_f"])
    assert ctx.md.r_f is not None


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


def test_writeback_spec_and_apply_for_ml():
    ctx = _make_ctx()
    adapter = _adapter()
    result = _run(adapter, ctx)
    ar = adapter.analyze(result, ctx)
    spec = adapter.get_writeback_spec(ar, ctx)
    keys = [item.key for item in spec]
    assert "readout_rf" in keys
    assert "ro_waveform" in keys

    adapter.apply_writeback(ctx, ar, ["readout_rf", "ro_waveform"])
    assert "readout_rf" in ctx.ml.modules
    assert "ro_waveform" in ctx.ml.waveforms


def test_writeback_spec_and_apply_for_ml_when_missing():
    ctx = _make_ctx()
    # ml is empty (no pre-registered modules/waveforms)
    adapter = _adapter()
    result = _run(adapter, ctx)
    ar = adapter.analyze(result, ctx)
    spec = adapter.get_writeback_spec(ar, ctx)
    keys = [item.key for item in spec]
    assert "readout_rf" in keys
    assert "ro_waveform" in keys

    adapter.apply_writeback(ctx, ar, ["readout_rf", "ro_waveform"])
    assert "readout_rf" in ctx.ml.modules
    assert "ro_waveform" in ctx.ml.waveforms


def test_writeback_edit_template_provided():
    """get_writeback_spec should provide edit_template for readout_rf and ro_waveform."""
    ctx = _make_ctx()
    adapter = _adapter()
    result = _run(adapter, ctx)
    ar = adapter.analyze(result, ctx)
    spec = adapter.get_writeback_spec(ar, ctx)

    spec_by_key = {item.key: item for item in spec}
    assert spec_by_key["readout_rf"].edit_template is not None
    assert spec_by_key["ro_waveform"].edit_template is not None
    assert isinstance(spec_by_key["readout_rf"].edit_template, CfgSchema)
    assert isinstance(spec_by_key["ro_waveform"].edit_template, CfgSchema)


def test_apply_writeback_with_overrides():
    """apply_writeback should use raw dict override instead of auto-build."""
    ctx = _make_ctx()
    adapter = _adapter()
    result = _run(adapter, ctx)
    ar = adapter.analyze(result, ctx)

    override_readout = {
        "type": "readout/pulse",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": 2.0},
            "ch": 1,
            "nqz": 2,
            "freq": ar.freq,
            "gain": 0.5,
        },
        "ro_cfg": {
            "ro_ch": 1,
            "ro_length": 1.8,
            "trig_offset": 0.4,
        },
    }

    adapter.apply_writeback(
        ctx,
        ar,
        ["readout_rf"],
        overrides={"readout_rf": {"name": "readout_rf", "cfg": override_readout}},
    )
    assert "readout_rf" in ctx.ml.modules


def test_no_last_cfg_side_channel():
    """FakeFreqAdapter must not set last_cfg as an instance attribute after run."""
    ctx = _make_ctx()
    adapter = _adapter()
    _run(adapter, ctx)
    assert "last_cfg" not in adapter.__dict__
