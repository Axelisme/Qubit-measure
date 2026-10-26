"""CSV input contracts at the public T2 load and preparation stages."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from zcu_tools.notebook.analysis.fit_tools import predict_f01_mhz
from zcu_tools.notebook.analysis.t2_curve import (
    T2CurveAnalysisConfig,
    calibrate_t2_flux,
    load_t2_curve_context,
    plot_t2_dephasing_data,
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


@pytest.mark.parametrize("csv_flux", [None, [0.8]])
def test_load_csv_requires_only_used_columns(
    sample_dir: Path, csv_flux: list[float] | None
) -> None:
    frame = pd.DataFrame(
        {
            "t1_us": [40],
            "t2e_us": [20],
            "unused": [np.nan],
            "t2r_err_us": ["unused error metadata"],
        }
    )
    if csv_flux is not None:
        frame["flux"] = csv_flux
    frame.to_csv(sample_dir / "samples.csv", index=False)
    context = load_t2_curve_context(result_dir=str(sample_dir), fluxs=np.array([0.51]))

    for column in ("freq_mhz", "t1_err_us", "t2e_err_us", "t2r_us"):
        assert np.isnan(context.samples_df[column]).all()
    data = prepare_t2_dephasing_data(calibrate_t2_flux(context))
    assert data.fit.fluxs.tolist() == [0.51]
    assert data.fit.T2e_us.tolist() == [20.0]
    assert np.isfinite(data.fit.f01_mhz).all()


def test_author_flux_is_not_automatically_corrected(sample_dir: Path) -> None:
    observed = predict_f01_mhz((3.4, 0.9, 0.6), np.array([0.53]))
    pd.DataFrame(
        {"flux": [0.8], "freq_mhz": observed, "t1_us": [40], "t2e_us": [20]}
    ).to_csv(sample_dir / "samples.csv", index=False)
    context = load_t2_curve_context(result_dir=str(sample_dir), fluxs=np.array([0.52]))
    data = prepare_t2_dephasing_data(
        calibrate_t2_flux(context), analysis_flux_range=(0.49, 0.55)
    )
    np.testing.assert_allclose(data.window.raw_fluxs, [0.52])
    np.testing.assert_allclose(data.window.fluxs, [0.52])


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
            fluxs=np.linspace(0.51, 0.53, 8),
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


@pytest.mark.parametrize(
    ("fluxs", "problem"),
    [
        (np.array([0.5, 0.51]), "row"),
        (np.array([[0.5]]), "one-dimensional"),
        (np.array([np.nan]), "finite"),
        (np.array([np.inf]), "finite"),
    ],
)
def test_load_rejects_invalid_author_flux_at_entry(
    sample_dir: Path, fluxs: NDArray[np.float64], problem: str
) -> None:
    pd.DataFrame({"t1_us": [40], "t2e_us": [20]}).to_csv(
        sample_dir / "samples.csv", index=False
    )
    with pytest.raises(ValueError, match=f"fluxs.*{problem}"):
        load_t2_curve_context(result_dir=str(sample_dir), fluxs=fluxs)


def test_author_flux_preserves_row_pairing_when_preparation_sorts(
    sample_dir: Path,
) -> None:
    pd.DataFrame(
        {"t1_us": [40, 50], "t2e_us": [20, 25], "flux": ["ignored", "also ignored"]}
    ).to_csv(sample_dir / "samples.csv", index=False)
    context = load_t2_curve_context(
        result_dir=str(sample_dir), fluxs=np.array([0.52, 0.51])
    )
    data = prepare_t2_dephasing_data(calibrate_t2_flux(context))
    np.testing.assert_allclose(data.fit.fluxs, [0.51, 0.52])
    assert data.fit.T2e_us.tolist() == [25.0, 20.0]


@pytest.mark.parametrize("column", ["t1_us", "t2e_us"])
def test_load_names_missing_measurement_columns(sample_dir: Path, column: str) -> None:
    frame = pd.DataFrame({"t1_us": [40], "t2e_us": [20], "unused": [np.nan]})
    frame.drop(columns=column).to_csv(sample_dir / "samples.csv", index=False)
    with pytest.raises(ValueError, match=column):
        load_t2_curve_context(result_dir=str(sample_dir), fluxs=np.array([0.51]))


@pytest.mark.parametrize("column", ["t1_us", "t2e_us"])
def test_load_names_nonnumeric_used_measurement_columns(
    sample_dir: Path, column: str
) -> None:
    frame = pd.DataFrame({"t1_us": [40], "t2e_us": [20]})
    frame[column] = "invalid"
    frame.to_csv(sample_dir / "samples.csv", index=False)
    with pytest.raises(ValueError, match=f"{column}.*numeric"):
        load_t2_curve_context(result_dir=str(sample_dir), fluxs=np.array([0.51]))


def test_model_energies_and_author_flux_are_enough_to_plot(sample_dir: Path) -> None:
    (sample_dir / "params.json").write_text(
        '{"fluxdep_fit": {"params": {"EJ": 3.4, "EC": 0.9, "EL": 0.6}}}'
    )
    pd.DataFrame({"t1_us": [40, 50], "t2e_us": [20, 25]}).to_csv(
        sample_dir / "samples.csv", index=False
    )
    context = load_t2_curve_context(
        result_dir=str(sample_dir), fluxs=np.array([0.51, 0.52])
    )
    data = prepare_t2_dephasing_data(calibrate_t2_flux(context))
    fig, ax = plot_t2_dephasing_data(data)
    try:
        np.testing.assert_allclose(
            np.asarray(ax.get_lines()[0].get_xdata(), dtype=np.float64), [0.51, 0.52]
        )
    finally:
        plt.close(fig)
