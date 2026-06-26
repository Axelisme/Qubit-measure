from __future__ import annotations

import json

import numpy as np
import pytest
from zcu_tools.meta_tool import (
    ArbWaveformData,
    ArbWaveformDatabase,
    ArbWaveformError,
    ArbWaveformPreview,
    FormulaRecipe,
    prepare_preview_series,
    render_formula_recipe,
)


def _init_db(tmp_path):
    root = tmp_path / "arb_waveforms"
    ArbWaveformDatabase.init(root)
    return root


def test_import_data_uses_single_file_layout_and_runtime_get(tmp_path):
    root = _init_db(tmp_path)

    info = ArbWaveformDatabase.import_data(
        "arb_data1",
        idata=np.array([0.0, 0.5, 0.0]),
        qdata=None,
        time=np.array([0.0, 0.25, 0.75]),
    )

    assert info.data_key == "arb_data1"
    assert info.duration == 0.75
    assert (root / "arb_data1.npz").exists()
    assert not (root / "arb_data1").exists()
    assert ArbWaveformDatabase.list() == ["arb_data1"]
    idata, qdata, time = ArbWaveformDatabase.get("arb_data1")
    assert np.allclose(idata, [0.0, 0.5, 0.0])
    assert qdata is None
    assert np.allclose(time, [0.0, 0.25, 0.75])


def test_import_data_rejects_invalid_key_before_filesystem(tmp_path):
    _init_db(tmp_path)

    with pytest.raises(ArbWaveformError) as exc:
        ArbWaveformDatabase.import_data(
            "../bad",
            idata=np.array([0.0, 0.0]),
            qdata=None,
            time=np.array([0.0, 1.0]),
        )

    assert exc.value.reason == "invalid_data_key"


def test_import_data_accepts_nonuniform_strictly_increasing_time(tmp_path):
    _init_db(tmp_path)

    info = ArbWaveformDatabase.import_data(
        "raw_nonuniform",
        idata=np.array([0.0, 0.2, 0.1, 0.0]),
        qdata=np.array([0.0, 0.1, 0.2, 0.0]),
        time=np.array([0.0, 0.1, 0.15, 0.9]),
    )

    assert info.sample_count == 4
    assert info.peak_abs == pytest.approx(np.hypot(0.2, 0.1))


def test_npz_unknown_key_and_invalid_recipe_json_fail_fast(tmp_path):
    root = _init_db(tmp_path)
    np.savez(
        root / "bad_extra.npz",
        idata=np.array([0.0, 0.0]),
        qdata=np.array([0.0, 0.0]),
        time=np.array([0.0, 1.0]),
        gain=np.array([1.0]),
    )
    np.savez(
        root / "bad_recipe.npz",
        idata=np.array([0.0, 0.0]),
        qdata=np.array([0.0, 0.0]),
        time=np.array([0.0, 1.0]),
        recipe_json=np.array('{"segments": []}'),
    )

    with pytest.raises(ArbWaveformError) as extra_exc:
        ArbWaveformDatabase.load("bad_extra")
    with pytest.raises(ArbWaveformError) as recipe_exc:
        ArbWaveformDatabase.load("bad_recipe")

    assert extra_exc.value.reason == "unknown_npz_key"
    assert recipe_exc.value.reason == "invalid_recipe_json"


def test_create_from_formula_embeds_loadable_recipe_and_preserves_formula_text(
    tmp_path,
):
    root = _init_db(tmp_path)
    recipe = {
        "segments": [{"duration": 1.0, "formula": " sin(2*pi*t) "}],
        "normalize": "none",
    }

    info = ArbWaveformDatabase.create_from_formula("formula1", recipe)

    assert info.sample_count == 1001
    assert info.duration == pytest.approx(1.0)
    assert info.has_recipe is True
    loaded_recipe = ArbWaveformDatabase.load_recipe("formula1")
    assert loaded_recipe is not None
    assert loaded_recipe.segments[0].formula == " sin(2*pi*t) "
    with np.load(root / "formula1.npz", allow_pickle=False) as archive:
        saved = json.loads(str(archive["recipe_json"].item()))
    assert saved == recipe


def test_formula_sample_count_and_non_grid_segment_durations():
    data = render_formula_recipe(
        {
            "segments": [
                {"duration": 0.0011, "formula": "0"},
                {"duration": 0.0012, "formula": "1"},
            ],
            "normalize": "none",
        }
    )

    assert data.time.size == round(0.0023 * 1000) + 1
    assert np.allclose(data.time, [0.0, 0.00115, 0.0023])
    assert np.allclose(data.idata, [0.0, 1.0, 1.0])


def test_formula_exact_internal_boundary_uses_following_segment():
    data = render_formula_recipe(
        {
            "segments": [
                {"duration": 0.001, "formula": "0"},
                {"duration": 0.001, "formula": "1"},
            ],
            "normalize": "none",
        }
    )

    assert np.allclose(data.time, [0.0, 0.001, 0.002])
    assert np.allclose(data.idata, [0.0, 1.0, 1.0])


def test_complex_formula_maps_real_and_imaginary_to_iq():
    data = render_formula_recipe(
        {
            "segments": [{"duration": 0.002, "formula": "0.25 + 0.5*I"}],
            "normalize": "none",
        }
    )

    assert np.allclose(data.idata, [0.25, 0.25, 0.25])
    assert np.allclose(data.qdata, [0.5, 0.5, 0.5])
    assert data.peak_abs == pytest.approx(np.hypot(0.25, 0.5))


def test_peak_normalization_uses_magnitude_and_amplifies_subunit_waveform():
    data = render_formula_recipe(
        {
            "segments": [{"duration": 0.002, "formula": "0.3 + 0.4*I"}],
            "normalize": "peak",
        }
    )

    assert np.allclose(data.idata, [0.6, 0.6, 0.6])
    assert np.allclose(data.qdata, [0.8, 0.8, 0.8])
    assert data.peak_abs == pytest.approx(1.0)


def test_normalize_none_rejects_magnitude_over_one():
    with pytest.raises(ArbWaveformError) as exc:
        render_formula_recipe(
            {
                "segments": [{"duration": 0.002, "formula": "0.8 + 0.8*I"}],
                "normalize": "none",
            }
        )

    assert exc.value.reason == "amplitude_out_of_range"


def test_formula_rejects_unsupported_syntax_and_names():
    for formula, reason in [
        ("Piecewise((1, t > 0), (0, True))", "formula_syntax_not_supported"),
        ("where(t > 0, 1, 0)", "formula_syntax_not_supported"),
        ("t.__class__", "formula_syntax_not_supported"),
        ("foo(t)", "formula_unknown_function"),
    ]:
        with pytest.raises(ArbWaveformError) as exc:
            render_formula_recipe(
                {
                    "segments": [{"duration": 0.002, "formula": formula}],
                    "normalize": "none",
                }
            )
        assert exc.value.reason == reason


def test_collision_update_delete_and_rename_policy(tmp_path):
    _init_db(tmp_path)
    recipe = FormulaRecipe.from_raw(
        {
            "segments": [{"duration": 0.002, "formula": "0"}],
            "normalize": "none",
        }
    )

    ArbWaveformDatabase.create_from_formula("asset_a", recipe)
    with pytest.raises(ArbWaveformError) as collision_exc:
        ArbWaveformDatabase.create_from_formula("asset_a", recipe)
    with pytest.raises(ArbWaveformError) as missing_update_exc:
        ArbWaveformDatabase.update_formula("missing", recipe)

    ArbWaveformDatabase.create_from_formula("asset_b", recipe)
    with pytest.raises(ArbWaveformError) as rename_collision_exc:
        ArbWaveformDatabase.rename("asset_a", "asset_b")
    ArbWaveformDatabase.rename("asset_a", "asset_c")
    ArbWaveformDatabase.delete("asset_c")

    assert collision_exc.value.reason == "data_key_exists"
    assert missing_update_exc.value.reason == "data_key_not_found"
    assert rename_collision_exc.value.reason == "data_key_exists"
    assert ArbWaveformDatabase.list() == ["asset_b"]


# ---------------------------------------------------------------------------
# D10. prepare_preview_series unit tests
# ---------------------------------------------------------------------------


def _make_data(i: float, q: float, duration: float = 0.002) -> ArbWaveformData:
    """Helper: render a constant I+Q waveform via formula."""
    return render_formula_recipe(
        {
            "segments": [{"duration": duration, "formula": f"{i} + {q}*I"}],
            "normalize": "none",
        }
    )


def test_D10_prepare_preview_series_normalize_true_divides_by_peak():
    """normalize=True: I/Q are divided by the peak |IQ|; abs_data == hypot(i, q)."""
    data = _make_data(i=0.3, q=0.4)
    expected_peak = float(np.hypot(0.3, 0.4))

    series = prepare_preview_series(data, normalize=True)

    assert isinstance(series, ArbWaveformPreview)
    assert np.allclose(series.idata, np.asarray(data.idata) / expected_peak)
    assert np.allclose(series.qdata, np.asarray(data.qdata) / expected_peak)
    assert np.allclose(series.abs_data, np.hypot(series.idata, series.qdata))
    assert np.allclose(series.time, data.time)


def test_D10_prepare_preview_series_normalize_false_keeps_raw():
    """normalize=False: I/Q are unchanged; abs_data == hypot(i, q)."""
    data = _make_data(i=0.3, q=0.4)

    series = prepare_preview_series(data, normalize=False)

    assert np.allclose(series.idata, data.idata)
    assert np.allclose(series.qdata, data.qdata)
    assert np.allclose(series.abs_data, np.hypot(data.idata, data.qdata))


def test_D10_prepare_preview_series_all_zero_no_division_error():
    """All-zero waveform with normalize=True must not raise a ZeroDivisionError."""
    data = render_formula_recipe(
        {
            "segments": [{"duration": 0.002, "formula": "0"}],
            "normalize": "none",
        }
    )

    series = prepare_preview_series(data, normalize=True)

    # Peak is 0 → no division; raw (all-zero) data returned unchanged.
    assert np.all(series.idata == 0.0)
    assert np.all(series.qdata == 0.0)
    assert np.all(series.abs_data == 0.0)
