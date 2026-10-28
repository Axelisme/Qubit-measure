"""CSV inputs and author coordinates at the public T1 stages."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from zcu_tools.notebook.analysis.fit_tools import predict_f01_mhz
from zcu_tools.notebook.analysis.t1_curve import (
    calibrate_t1_flux,
    load_t1_curve_context,
    plot_t1_curve_data,
    prepare_t1_curve_data,
    save_t1_curve_figure,
)
from zcu_tools.resources.qubit_params import FluxDepFit, QubitParams


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    QubitParams(tmp_path / "params.json").set_fluxdep_fit(
        FluxDepFit(EJ=3.4, EC=0.9, EL=0.6, flux_half=0, flux_int=0.5, flux_period=1)
    )
    return tmp_path


@pytest.mark.parametrize("custom_images", [False, True])
def test_load_leaves_input_tree_unchanged(
    sample_dir: Path, custom_images: bool
) -> None:
    pd.DataFrame({"t1_us": [40]}).to_csv(sample_dir / "samples.csv", index=False)
    before = set(sample_dir.rglob("*"))
    images = sample_dir / "nested" / "images" if custom_images else None
    context = load_t1_curve_context(
        result_dir=str(sample_dir),
        fluxs=np.array([0.51]),
        image_dir=str(images) if images else None,
    )
    assert context.image_dir == str(images or sample_dir / "t1_curve")
    assert set(sample_dir.rglob("*")) == before


def test_save_creates_output_directory_only_when_writing(sample_dir: Path) -> None:
    pd.DataFrame({"t1_us": [40]}).to_csv(sample_dir / "samples.csv", index=False)
    context = load_t1_curve_context(result_dir=str(sample_dir), fluxs=np.array([0.51]))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 2])
    try:
        saved = save_t1_curve_figure(fig, context, "curve.png", show=False)
        assert Path(saved) == sample_dir / "t1_curve" / "curve.png"
        assert Path(saved).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(fig)


@pytest.mark.parametrize("csv_flux", [None, [0.8]])
def test_load_csv_requires_only_used_columns(
    sample_dir: Path, csv_flux: list[float] | None
) -> None:
    frame = pd.DataFrame({"t1_us": [40], "unused": [np.nan]})
    if csv_flux is not None:
        frame["flux"] = csv_flux
    frame.to_csv(sample_dir / "samples.csv", index=False)
    context = load_t1_curve_context(result_dir=str(sample_dir), fluxs=np.array([0.51]))

    assert context.samples_df["t1_us"].tolist() == [40.0]
    assert np.isnan(context.samples_df["freq_mhz"]).all()
    assert np.isnan(context.samples_df["t1_err_us"]).all()
    data = prepare_t1_curve_data(calibrate_t1_flux(context))
    assert data.fit.fluxs.tolist() == [0.51]
    assert data.fit.T1_ns.tolist() == [40_000.0]
    assert np.isfinite(data.fit.f01_mhz).all()
    assert not data.fit.f01_measured.any()


def test_author_flux_is_not_automatically_corrected(sample_dir: Path) -> None:
    observed = predict_f01_mhz((3.4, 0.9, 0.6), np.array([0.53]))
    pd.DataFrame({"flux": [0.8], "freq_mhz": observed, "t1_us": [40]}).to_csv(
        sample_dir / "samples.csv", index=False
    )
    context = load_t1_curve_context(result_dir=str(sample_dir), fluxs=np.array([0.52]))
    data = prepare_t1_curve_data(
        calibrate_t1_flux(context), analysis_flux_range=(0.49, 0.55)
    )
    np.testing.assert_allclose(data.sample.raw_fluxs, [0.52])
    np.testing.assert_allclose(data.sample.fluxs, [0.52])


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
    pd.DataFrame({"t1_us": [40]}).to_csv(sample_dir / "samples.csv", index=False)
    with pytest.raises(ValueError, match=f"fluxs.*{problem}"):
        load_t1_curve_context(result_dir=str(sample_dir), fluxs=fluxs)


def test_author_flux_preserves_row_pairing_when_preparation_sorts(
    sample_dir: Path,
) -> None:
    pd.DataFrame({"t1_us": [40, 50], "flux": ["ignored", "also ignored"]}).to_csv(
        sample_dir / "samples.csv", index=False
    )
    context = load_t1_curve_context(
        result_dir=str(sample_dir), fluxs=np.array([0.52, 0.51])
    )
    data = prepare_t1_curve_data(calibrate_t1_flux(context))
    np.testing.assert_allclose(data.fit.fluxs, [0.51, 0.52])
    assert data.fit.T1_ns.tolist() == [50_000.0, 40_000.0]


@pytest.mark.parametrize("column", ["t1_us"])
def test_load_names_missing_measurement_columns(sample_dir: Path, column: str) -> None:
    frame = pd.DataFrame({"t1_us": [40], "unused": [np.nan]})
    frame.drop(columns=column).to_csv(sample_dir / "samples.csv", index=False)
    with pytest.raises(ValueError, match=column):
        load_t1_curve_context(result_dir=str(sample_dir), fluxs=np.array([0.51]))


@pytest.mark.parametrize("column", ["t1_us"])
def test_load_names_nonnumeric_used_measurement_columns(
    sample_dir: Path, column: str
) -> None:
    frame = pd.DataFrame({"t1_us": [40]})
    frame[column] = "invalid"
    frame.to_csv(sample_dir / "samples.csv", index=False)
    with pytest.raises(ValueError, match=f"{column}.*numeric"):
        load_t1_curve_context(result_dir=str(sample_dir), fluxs=np.array([0.51]))


def test_model_energies_and_author_flux_are_enough_to_plot(sample_dir: Path) -> None:
    (sample_dir / "params.json").write_text(
        '{"fluxdep_fit": {"params": {"EJ": 3.4, "EC": 0.9, "EL": 0.6}}}'
    )
    pd.DataFrame({"t1_us": [40, 50]}).to_csv(sample_dir / "samples.csv", index=False)
    context = load_t1_curve_context(
        result_dir=str(sample_dir), fluxs=np.array([0.51, 0.52])
    )
    data = prepare_t1_curve_data(calibrate_t1_flux(context))
    fig, ax = plot_t1_curve_data(data)
    try:
        np.testing.assert_allclose(
            np.asarray(ax.get_lines()[0].get_xdata(), dtype=np.float64), [0.51, 0.52]
        )
    finally:
        plt.close(fig)
