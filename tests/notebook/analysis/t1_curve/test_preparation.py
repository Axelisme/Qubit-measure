"""CSV inputs and calibrated coordinates at the public T1 stages."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zcu_tools.notebook.analysis.fit_tools import predict_f01_mhz
from zcu_tools.notebook.analysis.t1_curve import (
    calibrate_t1_flux,
    load_t1_curve_context,
    prepare_t1_curve_data,
)
from zcu_tools.resources.qubit_params import FluxDepFit, QubitParams


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    QubitParams(tmp_path / "params.json").set_fluxdep_fit(
        FluxDepFit(EJ=3.4, EC=0.9, EL=0.6, flux_half=0, flux_int=0.5, flux_period=1)
    )
    return tmp_path


def test_load_csv_requires_only_used_columns(sample_dir: Path) -> None:
    pd.DataFrame({"flux": [0.51], "t1_us": [40], "unused": [np.nan]}).to_csv(
        sample_dir / "samples.csv", index=False
    )
    context = load_t1_curve_context(result_dir=str(sample_dir))

    assert context.samples_df["flux"].tolist() == [0.51]
    assert context.samples_df["t1_us"].tolist() == [40.0]
    assert np.isnan(context.samples_df["freq_mhz"]).all()
    assert np.isnan(context.samples_df["t1_err_us"]).all()
    data = prepare_t1_curve_data(calibrate_t1_flux(context))
    assert data.fit.T1_ns.tolist() == [40_000.0]
    assert np.isfinite(data.fit.f01_mhz).all()
    assert not data.fit.f01_measured.any()


@pytest.mark.parametrize("enabled", [True, False])
def test_csv_flux_is_the_initial_coordinate_for_f01_correction(
    sample_dir: Path, enabled: bool
) -> None:
    observed = predict_f01_mhz((3.4, 0.9, 0.6), np.array([0.53]))
    pd.DataFrame(
        {"flux": [0.52], "freq_mhz": observed, "t1_us": [40]}
    ).to_csv(sample_dir / "samples.csv", index=False)
    context = load_t1_curve_context(result_dir=str(sample_dir))
    data = prepare_t1_curve_data(
        calibrate_t1_flux(context),
        analysis_flux_range=(0.49, 0.55),
        correct_flux_from_f01_enabled=enabled,
    )
    coordinates = data.sample
    np.testing.assert_allclose(coordinates.raw_fluxs, [0.52])
    np.testing.assert_allclose(coordinates.fluxs, [0.53 if enabled else 0.52], atol=1e-5)
    assert coordinates.f01_correction_applied.tolist() == [enabled]
