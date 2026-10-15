from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.resources.waveform_assets import (
    ArbWaveformData,
    ArbWaveformDatabase,
    ArbWaveformError,
    ArbWaveformPreview,
    FormulaRecipe,
    prepare_preview_series,
    render_formula_recipe,
)


def _save_zero_q_asset(
    repository: ArbWaveformDatabase, data_key: str = "pulse"
) -> None:
    repository.save(data_key, idata=np.array([0.0, 0.4]), time=np.array([0.0, 0.75]))


def test_root_bound_repositories_isolate_interleaved_writes_and_captures(
    tmp_path: Path,
) -> None:
    try:
        first = ArbWaveformDatabase(str(tmp_path / "first"))
        second = ArbWaveformDatabase(tmp_path / "second")
    except TypeError as exc:
        pytest.fail(f"Repositories must accept and own independent roots: {exc}")

    first.save("pulse", idata=[0.0, 0.4], time=[0.0, 0.75])
    captured = first.bound_info()
    captured_path = captured.path_for("pulse")
    second.save("pulse", idata=[0.0, 0.8], time=[0.0, 2.0])
    assert first.list() == second.list() == ["pulse"]
    np.testing.assert_array_equal(first.load("pulse").idata, [0.0, 0.4])
    np.testing.assert_array_equal(second.load("pulse").idata, [0.0, 0.8])
    assert captured_path == (tmp_path / "first" / "pulse.npz").resolve()

    first.save("pulse", idata=[0.0, 0.3], time=[0.0, 1.5])
    assert captured.path_for("pulse") == captured_path
    assert captured.duration("pulse") == 1.5
    np.testing.assert_array_equal(captured.samples("pulse").idata, [0.0, 0.3])
    assert second.bound_info().duration("pulse") == 2.0

    second.rename("pulse", "renamed")
    assert first.list() == ["pulse"]
    assert second.list() == ["renamed"]
    first.delete("pulse")
    assert first.list() == []
    np.testing.assert_array_equal(second.load("renamed").idata, [0.0, 0.8])


def test_import_data_uses_single_file_layout_and_runtime_get(tmp_path):
    root = tmp_path / "arb_waveforms"
    repository = ArbWaveformDatabase(root)

    info = repository.import_data(
        "arb_data1",
        idata=np.array([0.0, 0.5, 0.0]),
        qdata=None,
        time=np.array([0.0, 0.25, 0.75]),
    )

    assert info.data_key == "arb_data1"
    assert info.duration == 0.75
    assert (root / "arb_data1.npz").exists()
    assert not (root / "arb_data1").exists()
    assert repository.list() == ["arb_data1"]
    idata, qdata, time = repository.get("arb_data1")
    assert np.allclose(idata, [0.0, 0.5, 0.0])
    assert qdata is None
    assert np.allclose(time, [0.0, 0.25, 0.75])


def test_info_at_binds_explicit_scope_without_switching_repository(
    tmp_path: Path,
) -> None:
    first = tmp_path / "arb_waveforms"
    repository = ArbWaveformDatabase(first)
    _save_zero_q_asset(repository)
    other = ArbWaveformDatabase(tmp_path / "second")
    other.save("pulse", idata=np.array([0.0, 0.8]), time=np.array([0.0, 2.0]))

    bound = ArbWaveformDatabase.info_at(first)
    assert bound.duration("pulse") == 0.75
    np.testing.assert_array_equal(bound.samples("pulse").time, [0.0, 0.75])
    assert other.bound_info().duration("pulse") == 2.0


def test_bound_info_keeps_scope_after_repository_switch(tmp_path: Path) -> None:
    repository = ArbWaveformDatabase(tmp_path / "arb_waveforms")
    repository.save(
        "pulse",
        idata=np.array([0.0, 0.4, 0.0]),
        qdata=np.array([0.0, 0.3, 0.0]),
        time=np.array([0.0, 0.25, 0.75]),
    )
    bound = repository.bound_info()

    other = ArbWaveformDatabase(tmp_path / "other_assets")
    other.save("pulse", idata=np.array([0.0, 0.8]), time=np.array([0.0, 2.0]))

    assert bound.duration("pulse") == 0.75
    samples = bound.samples("pulse")
    np.testing.assert_array_equal(samples.time, [0.0, 0.25, 0.75])
    np.testing.assert_array_equal(samples.idata, [0.0, 0.4, 0.0])
    np.testing.assert_array_equal(samples.qdata, [0.0, 0.3, 0.0])
    assert other.bound_info().duration("pulse") == 2.0
    assert other.bound_info().samples("pulse").qdata is None


def test_bound_info_reads_current_bytes_and_isolates_arrays(tmp_path: Path) -> None:
    root = tmp_path / "arb_waveforms"
    repository = ArbWaveformDatabase(root)
    _save_zero_q_asset(repository)
    bound = repository.bound_info()
    copied = deepcopy(bound)
    first = bound.samples("pulse")
    first.idata[1] = -0.9
    first.time[1] = 9.0
    np.testing.assert_array_equal(copied.samples("pulse").idata, [0.0, 0.4])
    assert copied.duration("pulse") == 0.75

    repository.save(
        "pulse",
        idata=np.array([0.0, 0.3]),
        qdata=np.array([0.0, 0.4]),
        time=np.array([0.0, 1.5]),
    )
    current = bound.samples("pulse")
    np.testing.assert_array_equal(current.idata, [0.0, 0.3])
    np.testing.assert_array_equal(current.qdata, [0.0, 0.4])
    if current.qdata is None:
        pytest.fail("Nonzero Q samples must not be discarded")
    current.qdata[1] = -0.8
    np.testing.assert_array_equal(copied.samples("pulse").qdata, [0.0, 0.4])
    assert copied.duration("pulse") == 1.5
    np.testing.assert_array_equal(first.idata, [0.0, -0.9])

    (root / "pulse.npz").unlink()
    with pytest.raises(ArbWaveformError) as exc:
        copied.samples("pulse")
    assert exc.value.reason == "data_key_not_found"


def test_bound_path_keeps_scope_and_checks_existence_without_decoding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "arb_waveforms"
    repository = ArbWaveformDatabase(root)
    (root / "opaque.npz").write_bytes(b"not an archive")
    bound = repository.bound_info()
    assert bound.path_for("opaque") == (root / "opaque.npz").resolve()
    with pytest.raises(ArbWaveformError) as error:
        bound.samples("opaque")
    assert error.value.reason == "npz_read_failed"


@pytest.mark.parametrize(
    ("key", "reason"),
    [("absent", "data_key_not_found"), ("../escape", "invalid_data_key")],
)
def test_bound_path_rejects_missing_and_invalid_keys(
    tmp_path: Path, key: str, reason: str
) -> None:
    repository = ArbWaveformDatabase(tmp_path / "arb_waveforms")
    bound = repository.bound_info()
    with pytest.raises(ArbWaveformError) as error:
        bound.path_for(key)
    assert error.value.reason == reason


def test_repository_rejects_formula_semantics_not_just_json(tmp_path: Path) -> None:
    root = tmp_path / "arb_waveforms"
    repository = ArbWaveformDatabase(root)
    np.savez(
        root / "invalid_formula.npz",
        time=[0, 1],
        idata=[0, 0.5],
        qdata=[0, 0],
        recipe_json=np.array('{"future_recipe": true}'),
    )
    with pytest.raises(ArbWaveformError):
        repository.load("invalid_formula")


@pytest.mark.parametrize("operation", ["duration", "samples"])
def test_bound_info_reports_missing_key_in_bound_scope(
    tmp_path: Path, operation: str
) -> None:
    root = tmp_path / "arb_waveforms"
    repository = ArbWaveformDatabase(root)
    _save_zero_q_asset(repository, "present")
    bound = repository.bound_info()
    read = bound.duration if operation == "duration" else bound.samples
    with pytest.raises(ArbWaveformError) as exc:
        read("missing")
    assert exc.value.reason == "data_key_not_found"
    assert exc.value.data == {"data_key": "missing", "available": ["present"]}
    assert str(root / "missing.npz") in str(exc.value)


@pytest.mark.parametrize("operation", ["duration", "samples"])
def test_bound_info_rejects_invalid_keys(tmp_path: Path, operation: str) -> None:
    repository = ArbWaveformDatabase(tmp_path / "arb_waveforms")
    bound = repository.bound_info()
    read = bound.duration if operation == "duration" else bound.samples
    with pytest.raises(ArbWaveformError) as exc:
        read("../escape")
    assert exc.value.reason == "invalid_data_key"


@pytest.mark.parametrize("operation", ["duration", "samples"])
def test_bound_info_propagates_archive_validation(
    tmp_path: Path, operation: str
) -> None:
    root = tmp_path / "arb_waveforms"
    repository = ArbWaveformDatabase(root)
    np.savez(
        root / "invalid.npz",
        idata=np.array([0.0, 0.4]),
        qdata=np.array([0.0, 0.0]),
        time=np.array([0.0, 0.0]),
    )
    bound = repository.bound_info()
    read = bound.duration if operation == "duration" else bound.samples
    with pytest.raises(ArbWaveformError) as exc:
        read("invalid")
    assert exc.value.reason == "time_not_strictly_increasing"


@pytest.mark.parametrize("operation", ["duration", "samples"])
def test_bound_info_propagates_io_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    root = tmp_path / "arb_waveforms"
    repository = ArbWaveformDatabase(root)
    _save_zero_q_asset(repository)
    bound = repository.bound_info()

    def unreadable(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("asset read denied")

    # Inject an I/O denial independent of the process user's filesystem permissions.
    monkeypatch.setattr(np, "load", unreadable)
    read = bound.duration if operation == "duration" else bound.samples
    with pytest.raises(ArbWaveformError) as exc:
        read("pulse")
    assert exc.value.reason == "npz_read_failed"
    assert exc.value.data == {"path": str(root / "pulse.npz")}
    assert isinstance(exc.value.__cause__, PermissionError)


def test_import_data_rejects_invalid_key_before_filesystem(tmp_path):
    repository = ArbWaveformDatabase(tmp_path / "arb_waveforms")

    with pytest.raises(ArbWaveformError) as exc:
        repository.import_data(
            "../bad",
            idata=np.array([0.0, 0.0]),
            qdata=None,
            time=np.array([0.0, 1.0]),
        )

    assert exc.value.reason == "invalid_data_key"


def test_import_data_accepts_nonuniform_strictly_increasing_time(tmp_path):
    repository = ArbWaveformDatabase(tmp_path / "arb_waveforms")

    info = repository.import_data(
        "raw_nonuniform",
        idata=np.array([0.0, 0.2, 0.1, 0.0]),
        qdata=np.array([0.0, 0.1, 0.2, 0.0]),
        time=np.array([0.0, 0.1, 0.15, 0.9]),
    )

    assert info.sample_count == 4
    assert info.peak_abs == pytest.approx(np.hypot(0.2, 0.1))


def test_npz_unknown_key_and_invalid_recipe_json_fail_fast(tmp_path):
    root = tmp_path / "arb_waveforms"
    repository = ArbWaveformDatabase(root)
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
        repository.load("bad_extra")
    with pytest.raises(ArbWaveformError) as recipe_exc:
        repository.load("bad_recipe")

    assert extra_exc.value.reason == "unknown_npz_key"
    assert recipe_exc.value.reason == "invalid_recipe_json"


def test_create_from_formula_embeds_loadable_recipe_and_preserves_formula_text(
    tmp_path,
):
    root = tmp_path / "arb_waveforms"
    repository = ArbWaveformDatabase(root)
    recipe: dict[str, object] = {
        "segments": [{"duration": 1.0, "formula": " sin(2*pi*t) "}],
        "normalize": "none",
    }

    repository.create_from_formula("formula1", recipe)
    info = repository.inspect("formula1")

    assert info.sample_count == 1001
    assert info.duration == pytest.approx(1.0)
    assert info.has_recipe is True
    loaded_recipe = repository.load_recipe("formula1")
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
    repository = ArbWaveformDatabase(tmp_path / "arb_waveforms")
    recipe = FormulaRecipe.from_raw(
        {
            "segments": [{"duration": 0.002, "formula": "0"}],
            "normalize": "none",
        }
    )

    repository.create_from_formula("asset_a", recipe)
    with pytest.raises(ArbWaveformError) as collision_exc:
        repository.create_from_formula("asset_a", recipe)
    with pytest.raises(ArbWaveformError) as missing_update_exc:
        repository.update_formula("missing", recipe)

    repository.create_from_formula("asset_b", recipe)
    with pytest.raises(ArbWaveformError) as rename_collision_exc:
        repository.rename("asset_a", "asset_b")
    repository.rename("asset_a", "asset_c")
    repository.delete("asset_c")

    assert collision_exc.value.reason == "data_key_exists"
    assert missing_update_exc.value.reason == "data_key_not_found"
    assert rename_collision_exc.value.reason == "data_key_exists"
    assert repository.list() == ["asset_b"]


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
