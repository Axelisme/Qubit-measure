from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zcu_tools.notebook.analysis.fit_tools.sample_merge as sample_merge
from zcu_tools.notebook.analysis.fit_tools import (
    SampleSource,
    merge_sample_sources,
)
from zcu_tools.simulate import flux2value


def test_merge_sample_sources_maps_source_flux_to_target_frame(tmp_path: Path) -> None:
    target_dir = _write_params(
        tmp_path / "target",
        params=(3.5, 1.0, 0.6),
        flux_half=-11.1,
        flux_period=24.5,
    )
    source_dir = _write_params(
        tmp_path / "source",
        params=(3.4, 0.9, 0.5),
        flux_half=-0.0107,
        flux_period=0.0246,
    )
    fluxs = np.array([0.50, 0.55], dtype=np.float64)
    source_values_a = flux2value(fluxs, -0.0107, 0.0246)
    source_path = tmp_path / "source_samples.csv"
    pd.DataFrame(
        {
            "calibrated mA": source_values_a,
            "Freq (MHz)": [410.0, 430.0],
            "T1 (us)": [80.0, 70.0],
            "T2E": [4.0, 5.0],
            "T2R": [2.0, 3.0],
            "date": ["2026-07-08", "2026-07-08"],
        }
    ).to_csv(source_path, index=False)

    result = merge_sample_sources(
        target_result_dir=target_dir,
        sources=(
            SampleSource(
                path=source_path,
                unit="A",
                source_result_dir=source_dir,
                label="source_batch",
            ),
        ),
    )

    expected_target_values = flux2value(fluxs, -11.1, 24.5)
    np.testing.assert_allclose(result.merged["Flux"], fluxs)
    np.testing.assert_allclose(result.merged["calibrated mA"], expected_target_values)
    np.testing.assert_allclose(result.merged["T2e (us)"], [4.0, 5.0])
    np.testing.assert_allclose(result.merged["T2r (us)"], [2.0, 3.0])
    assert "T2E" not in result.merged.columns
    assert result.summary_table.loc[0, "unit"] == "A"
    assert result.summary_table.loc[0, "source"] == "source_batch"


def test_merge_sample_sources_uses_explicit_current_scale_to_source_frame(
    tmp_path: Path,
) -> None:
    target_dir = _write_params(
        tmp_path / "target",
        params=(3.5, 1.0, 0.6),
        flux_half=-11.1,
        flux_period=24.5,
    )
    fluxs = np.array([0.50, 0.55], dtype=np.float64)
    values_mA = flux2value(fluxs, -11.1, 24.5)
    source_path = tmp_path / "a_values_in_target_frame.csv"
    pd.DataFrame(
        {
            "calibrated mA": values_mA / 1000.0,
            "Freq (MHz)": [410.0, 430.0],
            "T1 (us)": [80.0, 70.0],
        }
    ).to_csv(source_path, index=False)

    result = merge_sample_sources(
        target_result_dir=target_dir,
        sources=(
            SampleSource(
                path=source_path,
                unit="A",
                label="scaled",
                current_scale_to_source_frame=1000.0,
            ),
        ),
    )

    np.testing.assert_allclose(result.merged["Flux"], fluxs)
    np.testing.assert_allclose(result.merged["calibrated mA"], values_mA)


def test_merge_sample_sources_fits_one_batch_flux_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = _write_params(
        tmp_path / "target",
        params=(3.5, 1.0, 0.6),
        flux_half=0.0,
        flux_period=1.0,
    )
    raw_fluxs = np.array([0.45, 0.50, 0.55], dtype=np.float64)
    true_offset = 0.014
    source_values = flux2value(raw_fluxs, 0.0, 1.0)
    source_path = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "calibrated mA": source_values,
            "Freq (MHz)": 1000.0 + 200.0 * (raw_fluxs + true_offset),
            "T1 (us)": [10.0, 10.0, 10.0],
        }
    ).to_csv(source_path, index=False)

    def _linear_f01(
        _params: tuple[float, float, float], fluxs: np.ndarray, *, cutoff: int = 40
    ) -> np.ndarray:
        return 1000.0 + 200.0 * np.asarray(fluxs, dtype=np.float64)

    monkeypatch.setattr(sample_merge, "predict_f01_mhz", _linear_f01)

    result = merge_sample_sources(
        target_result_dir=target_dir,
        sources=(
            SampleSource(
                path=source_path,
                label="batch",
                fit_batch_flux_offset=True,
                max_abs_batch_flux_offset=0.05,
            ),
        ),
    )

    fitted = float(result.summary_table.loc[0, "fitted_flux_offset"])
    assert fitted == pytest.approx(true_offset, abs=2e-5)
    np.testing.assert_allclose(
        result.merged["Flux"], raw_fluxs + true_offset, atol=2e-5
    )
    assert "target_residual_before_offset_MHz" in result.diagnostics.columns
    assert (
        "target_residual_median_abs_before_offset_MHz" in result.summary_table.columns
    )
    assert "source_residual_median_abs_after_offset_MHz" in result.summary_table.columns


def test_batch_flux_offset_can_use_target_model_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_params = (1.0, 1.0, 0.5)
    target_params = (2.0, 1.0, 0.5)
    target_dir = _write_params(
        tmp_path / "target",
        params=target_params,
        flux_half=0.0,
        flux_period=1.0,
    )
    source_dir = _write_params(
        tmp_path / "source",
        params=source_params,
        flux_half=0.0,
        flux_period=1.0,
    )
    raw_fluxs = np.array([0.45, 0.50, 0.55], dtype=np.float64)
    true_offset = -0.012
    source_values = flux2value(raw_fluxs, 0.0, 1.0)
    source_path = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "calibrated mA": source_values,
            "Freq (MHz)": 1000.0 + 200.0 * (raw_fluxs + true_offset),
            "T1 (us)": [10.0, 10.0, 10.0],
        }
    ).to_csv(source_path, index=False)

    def _reference_sensitive_f01(
        params: tuple[float, float, float], fluxs: np.ndarray, *, cutoff: int = 40
    ) -> np.ndarray:
        fluxs_arr = np.asarray(fluxs, dtype=np.float64)
        if params == target_params:
            return 1000.0 + 200.0 * fluxs_arr
        if params == source_params:
            return 800.0 + 200.0 * fluxs_arr
        raise AssertionError(f"unexpected params {params!r}")

    monkeypatch.setattr(sample_merge, "predict_f01_mhz", _reference_sensitive_f01)

    result = merge_sample_sources(
        target_result_dir=target_dir,
        sources=(
            SampleSource(
                path=source_path,
                source_result_dir=source_dir,
                label="target-referenced",
                fit_batch_flux_offset=True,
                batch_flux_offset_reference="target",
                batch_flux_offset_objective="median_abs",
                max_abs_batch_flux_offset=0.05,
            ),
        ),
    )

    assert result.summary_table.loc[0, "batch_flux_offset_reference"] == "target"
    assert result.summary_table.loc[0, "batch_flux_offset_objective"] == "median_abs"
    fitted = float(result.summary_table.loc[0, "fitted_flux_offset"])
    assert fitted == pytest.approx(true_offset, abs=2e-5)
    np.testing.assert_allclose(
        result.merged["Flux"], raw_fluxs + true_offset, atol=2e-5
    )


def test_merge_sample_sources_rejects_conflicting_t2e_aliases(tmp_path: Path) -> None:
    target_dir = _write_params(
        tmp_path / "target",
        params=(3.5, 1.0, 0.6),
        flux_half=0.0,
        flux_period=1.0,
    )
    source_path = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "calibrated mA": [0.0],
            "Freq (MHz)": [1000.0],
            "T2E": [4.0],
            "T2e (us)": [5.0],
        }
    ).to_csv(source_path, index=False)

    with pytest.raises(ValueError, match="conflicting aliases.*T2e"):
        merge_sample_sources(
            target_result_dir=target_dir,
            sources=(SampleSource(path=source_path, label="bad"),),
        )


def _write_params(
    result_dir: Path,
    *,
    params: tuple[float, float, float],
    flux_half: float,
    flux_period: float,
) -> Path:
    result_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "project": {
            "chip_name": result_dir.name,
            "qubit_name": "Q1",
            "resonator_name": "unknown",
        },
        "fluxdep_fit": {
            "params": {"EJ": params[0], "EC": params[1], "EL": params[2]},
            "flux_half": flux_half,
            "flux_int": flux_half + flux_period / 2.0,
            "flux_period": flux_period,
            "plot_transitions": {},
        },
    }
    (result_dir / "params.json").write_text(json.dumps(payload), encoding="utf-8")
    return result_dir
