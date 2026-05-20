"""Phase 9.5 tests — FakeFreqAdapter simulate + analyze flow."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from zcu_tools.experiment.v2_gui.adapters.onetone.freq import FakeFreqAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.adapter import CfgSchema, ExpContext, ScalarField
from zcu_tools.gui.registry import Registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=None,
        soccfg=None,
    )


def _adapter() -> FakeFreqAdapter:
    return FakeFreqAdapter()


def _run(adapter: FakeFreqAdapter, ctx: ExpContext):
    schema = adapter.make_default_cfg(ctx)
    schema.root.fields["fast_mode"] = ScalarField(
        value=True, label="Fast mode", type=bool
    )
    return adapter.run(ctx, schema)


# ---------------------------------------------------------------------------
# make_default_cfg
# ---------------------------------------------------------------------------


def test_make_default_cfg_returns_cfg_schema():
    from zcu_tools.gui.adapter import CfgSchema

    adapter = _adapter()
    schema = adapter.make_default_cfg(_make_ctx())
    assert isinstance(schema, CfgSchema)


def test_make_default_cfg_has_expected_fields():
    adapter = _adapter()
    schema = adapter.make_default_cfg(_make_ctx())
    fields = schema.root.fields
    for key in (
        "reps",
        "rounds",
        "freq",  # SweepField for frequency sweep
        "res_freq",  # HangerModel resonator frequency
        "Ql",
    ):
        assert key in fields, f"missing field: {key}"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_returns_tuple_of_arrays():
    ctx = _make_ctx()
    adapter = _adapter()
    freqs, signals = _run(adapter, ctx)
    assert isinstance(freqs, np.ndarray)
    assert isinstance(signals, np.ndarray)


def test_run_shapes_match():
    ctx = _make_ctx()
    adapter = _adapter()
    freqs, signals = _run(adapter, ctx)
    assert freqs.shape == signals.shape


def test_run_freq_range_matches_cfg():
    ctx = _make_ctx()
    adapter = _adapter()
    schema = adapter.make_default_cfg(ctx)
    # default SweepField: start=5800.0, stop=6200.0, expts=201
    freqs, _ = adapter.run(ctx, schema)
    assert abs(freqs[0] - 5800.0) < 1.0
    assert abs(freqs[-1] - 6200.0) < 1.0
    assert len(freqs) == 201


def test_run_signals_are_complex():
    ctx = _make_ctx()
    adapter = _adapter()
    _, signals = _run(adapter, ctx)
    assert np.iscomplexobj(signals)


def test_run_noise_decreases_with_more_rounds():
    """More rounds → lower noise (SNR ∝ sqrt(rounds)).

    Run with a large noise_scale so the noise variance dominates the signal.
    Compare std of raw amplitudes between 1 round and 100 rounds.  At 100
    rounds the theoretical reduction is 10×; asserting > 3× gives a robust
    lower bound even with an unlucky random seed.
    """
    ctx = _make_ctx()
    adapter = _adapter()

    def _schema_with(rounds: int) -> CfgSchema:
        s = adapter.make_default_cfg(ctx)
        s.root.fields["rounds"] = ScalarField(value=rounds, label="Rounds", type=int)
        # large noise_scale relative to |a0|=1 makes noise dominate
        s.root.fields["noise_scale"] = ScalarField(
            value=10.0, label="Noise scale", type=float
        )
        s.root.fields["fast_mode"] = ScalarField(
            value=True, label="Fast mode", type=bool
        )
        return s

    _, sig_low = adapter.run(ctx, _schema_with(1))
    _, sig_high = adapter.run(ctx, _schema_with(100))

    # std of amplitudes: noise-dominated → should decrease with more rounds
    noise_low = float(np.std(np.abs(sig_low)))
    noise_high = float(np.std(np.abs(sig_high)))
    assert noise_low > noise_high * 3  # theory ≈ 10×; 3× is a safe lower bound


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_returns_four_tuple():
    ctx = _make_ctx()
    adapter = _adapter()
    result = _run(adapter, ctx)
    analyze_result = adapter.analyze(result, ctx)
    assert len(analyze_result) == 4


def test_analyze_freq_close_to_true_value():
    """Fitted frequency should be within 5 MHz of the ground-truth 6000.0 MHz."""
    ctx = _make_ctx()
    adapter = _adapter()
    # Use high rounds to reduce noise and make fit reliable
    schema = adapter.make_default_cfg(ctx)
    schema.root.fields["rounds"] = ScalarField(value=200, label="Rounds", type=int)
    schema.root.fields["fast_mode"] = ScalarField(
        value=True, label="Fast mode", type=bool
    )
    result = adapter.run(ctx, schema)
    freq_fit, _, _, _ = adapter.analyze(result, ctx)
    assert abs(freq_fit - 6000.0) < 5.0


def test_analyze_produces_figure():
    from matplotlib.figure import Figure

    ctx = _make_ctx()
    adapter = _adapter()
    result = _run(adapter, ctx)
    _, _, _, fig = adapter.analyze(result, ctx)
    assert isinstance(fig, Figure)


def test_get_figure_returns_figure():
    from matplotlib.figure import Figure

    ctx = _make_ctx()
    adapter = _adapter()
    result = _run(adapter, ctx)
    analyze_result = adapter.analyze(result, ctx)
    fig = adapter.get_figure(analyze_result)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# writeback spec
# ---------------------------------------------------------------------------


def test_get_writeback_spec_has_r_f_and_rf_w():
    ctx = _make_ctx()
    adapter = _adapter()
    result = _run(adapter, ctx)
    analyze_result = adapter.analyze(result, ctx)
    spec = adapter.get_writeback_spec(analyze_result, ctx)
    keys = [item.key for item in spec]
    assert "r_f" in keys
    assert "rf_w" in keys


def test_apply_writeback_sets_md_r_f():
    ctx = _make_ctx()
    adapter = _adapter()
    result = _run(adapter, ctx)
    analyze_result = adapter.analyze(result, ctx)
    adapter.apply_writeback(ctx, analyze_result, ["r_f"])
    assert ctx.md.r_f is not None


# ---------------------------------------------------------------------------
# save paths / registry
# ---------------------------------------------------------------------------


def test_make_save_paths_returns_save_paths():
    from zcu_tools.gui.adapter import SavePaths

    ctx = _make_ctx()
    adapter = _adapter()
    paths = adapter.make_save_paths(ctx)
    assert isinstance(paths, SavePaths)
    assert paths.data_path
    assert paths.image_path


def test_registered_in_registry():
    registry = Registry()
    register_all(registry)
    assert registry.has("onetone/freq")
    adapter = registry.create("onetone/freq")
    assert isinstance(adapter, FakeFreqAdapter)


def test_make_default_cfg_has_mod_ref_field():
    from typing import cast
    from zcu_tools.gui.adapter import ModuleRefField, CfgSection

    ctx = _make_ctx()
    adapter = _adapter()
    schema = adapter.make_default_cfg(ctx)
    modules_sec = schema.root.fields["modules"]
    assert isinstance(modules_sec, CfgSection)
    assert "readout" in modules_sec.fields
    assert isinstance(modules_sec.fields["readout"], ModuleRefField)


def test_writeback_spec_and_apply_for_ml():
    from typing import cast

    ctx = _make_ctx()
    # Mock ml to contain readout_rf and ro_waveform
    mock_readout_rf = MagicMock()
    mock_readout_rf.pulse_cfg = MagicMock()
    mock_readout_rf.pulse_cfg.freq = 5900.0
    ctx.ml.modules = {"readout_rf": mock_readout_rf}

    mock_ro_waveform = MagicMock()
    mock_ro_waveform.length = 0.5
    ctx.ml.waveforms = {"ro_waveform": mock_ro_waveform}

    adapter = _adapter()
    result = _run(adapter, ctx)
    analyze_result = adapter.analyze(result, ctx)
    spec = adapter.get_writeback_spec(analyze_result, ctx)
    keys = [item.key for item in spec]
    assert "readout_rf" in keys
    assert "ro_waveform" in keys

    # Apply writeback
    # Should update the module and waveforms
    adapter.apply_writeback(ctx, analyze_result, ["readout_rf", "ro_waveform"])
    cast(MagicMock, ctx.ml.register_module).assert_called_once()
    cast(MagicMock, ctx.ml.register_waveform).assert_called_once()


def test_writeback_spec_and_apply_for_ml_when_missing():
    from typing import cast

    ctx = _make_ctx()
    # Mock ml to be empty
    ctx.ml.modules = {}
    ctx.ml.waveforms = {}

    adapter = _adapter()
    result = _run(adapter, ctx)
    analyze_result = adapter.analyze(result, ctx)
    spec = adapter.get_writeback_spec(analyze_result, ctx)
    keys = [item.key for item in spec]
    assert "readout_rf" in keys
    assert "ro_waveform" in keys

    # Apply writeback
    # Should call register_module and register_waveform
    adapter.apply_writeback(ctx, analyze_result, ["readout_rf", "ro_waveform"])
    cast(MagicMock, ctx.ml.register_module).assert_called_once()
    cast(MagicMock, ctx.ml.register_waveform).assert_called_once()
