"""CSV input contracts at the public T2 load and preparation stages."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zcu_tools.notebook.analysis.fit_tools import predict_f01_mhz
from zcu_tools.notebook.analysis.t2_curve import (
    T2CurveAnalysisConfig,
    calibrate_t2_flux,
    load_t2_curve_context,
    prepare_t2_dephasing_data,
    run_t2_curve_analysis,
)
from zcu_tools.resources.qubit_params import DispersiveFit, FluxDepFit, QubitParams


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    params = QubitParams(tmp_path / "params.json")
    params.set_fluxdep_fit(
        FluxDepFit(EJ=3.4, EC=0.9, EL=0.6, flux_half=0.5, flux_int=0, flux_period=1)
    )
    params.set_dispersive_fit(DispersiveFit(g=0.1, bare_rf=7))
    return tmp_path


def test_load_csv_requires_only_used_columns(sample_dir: Path) -> None:
    pd.DataFrame(
        {"flux": [0.51], "t1_us": [40], "t2e_us": [20], "unused": [np.nan]}
    ).to_csv(sample_dir / "samples.csv", index=False)
    context = load_t2_curve_context(result_dir=str(sample_dir))

    assert context.samples_df["flux"].tolist() == [0.51]
    for column in ("freq_mhz", "t1_err_us", "t2e_err_us", "t2r_us", "t2r_err_us"):
        assert np.isnan(context.samples_df[column]).all()
    data = prepare_t2_dephasing_data(calibrate_t2_flux(context))
    assert data.fit.T2e_us.tolist() == [20.0]
    assert np.isfinite(data.fit.f01_mhz).all()


@pytest.mark.parametrize("enabled", [True, False])
def test_csv_flux_is_the_initial_coordinate_for_f01_correction(
    sample_dir: Path, enabled: bool
) -> None:
    observed = predict_f01_mhz((3.4, 0.9, 0.6), np.array([0.53]))
    pd.DataFrame(
        {"flux": [0.52], "freq_mhz": observed, "t1_us": [40], "t2e_us": [20]}
    ).to_csv(sample_dir / "samples.csv", index=False)
    context = load_t2_curve_context(result_dir=str(sample_dir))
    data = prepare_t2_dephasing_data(
        calibrate_t2_flux(context),
        analysis_flux_range=(0.49, 0.55),
        correct_flux_from_f01_enabled=enabled,
    )
    coordinates = data.window
    np.testing.assert_allclose(coordinates.raw_fluxs, [0.52])
    np.testing.assert_allclose(
        coordinates.fluxs, [0.53 if enabled else 0.52], atol=1e-5
    )
    assert coordinates.f01_correction_applied.tolist() == [enabled]


def test_missing_dispersive_disables_photon_channel_and_completes_analysis(
    sample_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = sample_dir / "params.json"
    raw = json.loads(path.read_text())
    del raw["dispersive"]
    path.write_text(json.dumps(raw))
    pd.DataFrame(
        {"flux": np.linspace(0.51, 0.53, 8), "t1_us": 40, "t2e_us": 20}
    ).to_csv(sample_dir / "samples.csv", index=False)

    result = run_t2_curve_analysis(
        T2CurveAnalysisConfig(
            result_dir=str(sample_dir),
            t_flux_count=12,
            progress=False,
            save_figures=False,
            show_figures=False,
        )
    )

    assert result.combined_fit.fit_result.success
    assert result.combined_fit.fit_result.params.n_th is None
    assert result.photon_probe is None
    assert not result.context.photon_shot_noise_enabled
    assert "photon shot noise disabled" in capsys.readouterr().out
    for resolution in (
        result.combined_fit.fit_result.T1_error_resolution,
        result.combined_fit.fit_result.T2_error_resolution,
    ):
        assert resolution is not None
        assert resolution.nan_mask.all()
        assert np.isfinite(resolution.effective_errors).all()
