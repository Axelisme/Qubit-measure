"""Phase 9.5 tests — FakeFreqAdapter simulate + analyze flow."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from zcu_tools.experiment.v2_gui.adapters.onetone.freq import FakeFreqAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.adapter import ExpContext, ScalarField
from zcu_tools.gui.registry import Registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        em=MagicMock(),
        soc=None,
        soccfg=None,
    )


def _adapter() -> FakeFreqAdapter:
    return FakeFreqAdapter()


def _run(adapter: FakeFreqAdapter, ctx: ExpContext):
    schema = adapter.make_default_cfg(ctx)
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
        "freq_start",
        "freq_stop",
        "freq_expts",
        "freq",
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
    # default: 5.8 to 6.2 MHz, 201 points
    freqs, _ = adapter.run(ctx, schema)
    assert abs(freqs[0] - 5.8) < 0.01
    assert abs(freqs[-1] - 6.2) < 0.01
    assert len(freqs) == 201


def test_run_signals_are_complex():
    ctx = _make_ctx()
    adapter = _adapter()
    _, signals = _run(adapter, ctx)
    assert np.iscomplexobj(signals)


def test_run_noise_decreases_with_more_rounds():
    """More rounds → lower noise (SNR ∝ sqrt(rounds))."""
    ctx = _make_ctx()
    adapter = _adapter()

    schema_low = adapter.make_default_cfg(ctx)
    schema_low.root.fields["rounds"] = ScalarField(value=1, label="Rounds", type=int)
    _, sig_low = adapter.run(ctx, schema_low)

    schema_high = adapter.make_default_cfg(ctx)
    schema_high.root.fields["rounds"] = ScalarField(value=400, label="Rounds", type=int)
    _, sig_high = adapter.run(ctx, schema_high)

    # The ideal signal is the same; noise amplitude ∝ 1/sqrt(rounds)
    # so std of the residual from the mean should be larger for low rounds
    noise_low = np.std(np.abs(sig_low))
    noise_high = np.std(np.abs(sig_high))
    assert noise_low > noise_high


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
    """Fitted frequency should be within 0.05 MHz of the ground-truth 6.0 MHz."""
    ctx = _make_ctx()
    adapter = _adapter()
    # Use high rounds to reduce noise and make fit reliable
    schema = adapter.make_default_cfg(ctx)
    schema.root.fields["rounds"] = ScalarField(value=200, label="Rounds", type=int)
    result = adapter.run(ctx, schema)
    freq_fit, _, _, _ = adapter.analyze(result, ctx)
    assert abs(freq_fit - 6.0) < 0.05


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
