"""Tests for the flux-first v2 SampleMerge (fit_tools/sample_merge.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zcu_tools.notebook.analysis.fit_tools.sample_merge as sample_merge
from zcu_tools.meta_tool import (
    DEV_UNIT_COLUMN,
    DEV_VALUE_COLUMN,
    FLUX_COLUMN,
    FLUX_INT_COLUMN,
    FLUX_PERIOD_COLUMN,
    SAMPLE_COORDINATE_COLUMNS,
    SampleTableV2Error,
)
from zcu_tools.notebook.analysis.fit_tools import (
    FluxFrame,
    SampleSource,
    merge_sample_sources,
    write_merged_samples,
    write_sample_merge_report,
)


def _target_frame(
    *,
    label: str = "target",
    dev_unit: str = "A",
    flux_int: float = -11.1,
    flux_period: float = 24.5,
) -> FluxFrame:
    return FluxFrame(
        params=(3.5, 1.0, 0.6),
        dev_unit=dev_unit,  # type: ignore[arg-type]
        flux_int=flux_int,
        flux_period=flux_period,
        label=label,
    )


def _write_v2_csv(
    path: Path,
    *,
    dev_values: np.ndarray,
    dev_unit: str,
    flux: np.ndarray | None = None,
    flux_int: np.ndarray | float | None = None,
    flux_period: np.ndarray | float | None = None,
    measurements: dict[str, list[object]] | None = None,
) -> Path:
    frame: dict[str, object] = {
        DEV_VALUE_COLUMN: list(dev_values),
        DEV_UNIT_COLUMN: [dev_unit] * len(dev_values),
    }
    if flux is not None:
        frame[FLUX_COLUMN] = np.asarray(flux, dtype=np.float64).tolist()
    if flux_int is not None:
        int_arr = np.asarray(flux_int, dtype=np.float64)
        frame[FLUX_INT_COLUMN] = (
            int_arr.tolist()
            if int_arr.ndim == 1
            else [int_arr.item()] * len(dev_values)
        )
    if flux_period is not None:
        period_arr = np.asarray(flux_period, dtype=np.float64)
        frame[FLUX_PERIOD_COLUMN] = (
            period_arr.tolist()
            if period_arr.ndim == 1
            else [period_arr.item()] * len(dev_values)
        )
    if measurements:
        frame.update(measurements)
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def test_flux_frame_round_trip_scalar_and_array() -> None:
    frame = _target_frame()

    scalar_flux = frame.flux_from_dev_value(-11.1)
    assert isinstance(scalar_flux, float)
    assert scalar_flux == pytest.approx(0.0)
    assert frame.dev_value_from_flux(scalar_flux) == pytest.approx(-11.1)

    fluxs = np.array([0.5, 0.55, -0.25, 1.75], dtype=np.float64)
    values = frame.dev_value_from_flux(fluxs)
    assert isinstance(values, np.ndarray)
    np.testing.assert_allclose(frame.flux_from_dev_value(values), fluxs)
    np.testing.assert_allclose(values, fluxs * 24.5 + (-11.1))


def test_flux_frame_rejects_invalid_frames() -> None:
    with pytest.raises(ValueError, match="unsupported device unit"):
        FluxFrame(
            params=(1.0, 2.0, 3.0),
            dev_unit="mA",  # type: ignore[arg-type]
            flux_int=0.0,
            flux_period=1.0,
            label="bad",
        )
    with pytest.raises(ValueError, match="flux_int"):
        FluxFrame(
            params=(1.0, 2.0, 3.0),
            dev_unit="A",
            flux_int=np.nan,
            flux_period=1.0,
            label="bad",
        )
    with pytest.raises(ValueError, match="flux_period"):
        FluxFrame(
            params=(1.0, 2.0, 3.0),
            dev_unit="A",
            flux_int=0.0,
            flux_period=0.0,
            label="bad",
        )
    with pytest.raises(ValueError, match="flux_period"):
        FluxFrame(
            params=(1.0, 2.0, 3.0),
            dev_unit="A",
            flux_int=0.0,
            flux_period=-2.0,
            label="bad",
        )
    with pytest.raises(ValueError, match="flux_period"):
        FluxFrame(
            params=(1.0, 2.0, 3.0),
            dev_unit="A",
            flux_int=0.0,
            flux_period=np.inf,
            label="bad",
        )
    with pytest.raises(ValueError, match="params"):
        FluxFrame(
            params=(1.0, 2.0),  # type: ignore[arg-type]
            dev_unit="A",
            flux_int=0.0,
            flux_period=1.0,
            label="bad",
        )
    with pytest.raises(ValueError, match="params"):
        FluxFrame(
            params=(np.inf, 2.0, 3.0),
            dev_unit="A",
            flux_int=0.0,
            flux_period=1.0,
            label="bad",
        )


def test_from_result_dir_requires_explicit_dev_unit(tmp_path: Path) -> None:
    target_dir = _write_params(tmp_path / "target")

    with pytest.raises(TypeError):
        FluxFrame.from_result_dir(target_dir)  # type: ignore[call-arg]

    frame = FluxFrame.from_result_dir(target_dir, dev_unit="A")
    assert frame.dev_unit == "A"
    assert frame.flux_int == pytest.approx(1.15)
    assert frame.flux_period == pytest.approx(24.5)
    assert frame.params == (3.5, 1.0, 0.6)
    assert frame.label == str(target_dir)

    labeled = FluxFrame.from_result_dir(target_dir, dev_unit="V", label="custom")
    assert labeled.dev_unit == "V"
    assert labeled.label == "custom"


def test_merge_requires_exactly_one_target_identity(tmp_path: Path) -> None:
    target_dir = _write_params(tmp_path / "target")
    source_path = _write_v2_csv(
        tmp_path / "samples.csv",
        dev_values=np.array([1.0, 2.0]),
        dev_unit="A",
        flux=np.array([0.5, 0.55]),
    )
    sources = (SampleSource(path=source_path, label="s"),)

    with pytest.raises(TypeError):
        merge_sample_sources(sources=sources)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        merge_sample_sources(  # type: ignore[call-arg]
            target_result_dir=target_dir,  # type: ignore[call-arg]
            sources=sources,
            target_frame=_target_frame(),
        )
    with pytest.raises(ValueError, match="At least one SampleSource"):
        merge_sample_sources(target_frame=_target_frame(), sources=())


def test_merge_explicit_flux_provenance_and_target_output(tmp_path: Path) -> None:
    target = _target_frame()
    fluxs = np.array([0.50, 0.55], dtype=np.float64)
    source_values = target.dev_value_from_flux(fluxs)
    source_path = _write_v2_csv(
        tmp_path / "source_samples.csv",
        dev_values=np.asarray(source_values, dtype=np.float64),
        dev_unit="A",
        flux=fluxs,
        measurements={
            "Freq (MHz)": [410.0, 430.0],
            "T1 (us)": [80.0, 70.0],
            "T2E": [4.0, 5.0],
            "T2R": [2.0, 3.0],
            "date": ["2026-07-08", "2026-07-08"],
        },
    )

    result = merge_sample_sources(
        target_frame=target,
        sources=(SampleSource(path=source_path, label="source_batch"),),
    )

    merged = result.merged
    assert list(merged.columns[: len(SAMPLE_COORDINATE_COLUMNS)]) == list(
        SAMPLE_COORDINATE_COLUMNS
    )
    np.testing.assert_allclose(merged[FLUX_COLUMN], fluxs)
    np.testing.assert_allclose(merged[DEV_VALUE_COLUMN], source_values)
    assert (merged[DEV_UNIT_COLUMN] == "A").all()
    np.testing.assert_allclose(merged[FLUX_INT_COLUMN], target.flux_int)
    np.testing.assert_allclose(merged[FLUX_PERIOD_COLUMN], target.flux_period)
    # Caller-owned measurement columns pass through with their original names.
    np.testing.assert_allclose(merged["T1 (us)"], [80.0, 70.0])
    np.testing.assert_allclose(merged["T2E"], [4.0, 5.0])
    assert "T2e (us)" not in merged.columns
    assert list(merged["date"]) == ["2026-07-08", "2026-07-08"]

    assert list(result.diagnostics["flux_source"]) == ["explicit", "explicit"]
    assert (result.diagnostics["integer_flux_offset"] == 0).all()
    assert result.summary_table.loc[0, "source"] == "source_batch"
    assert result.summary_table.loc[0, "explicit_flux_rows"] == 2
    assert result.summary_table.loc[0, "row_frame_flux_rows"] == 0
    assert result.summary_table.loc[0, "fallback_frame_flux_rows"] == 0
    assert result.summary_table.loc[0, "target_frame"] == "target"


def test_merge_row_frame_provenance(tmp_path: Path) -> None:
    target = _target_frame()
    source_frame = FluxFrame(
        params=(3.4, 0.9, 0.5),
        dev_unit="A",
        flux_int=-0.0107,
        flux_period=0.0246,
        label="row_frame",
    )
    fluxs = np.array([0.50, 0.55], dtype=np.float64)
    source_values = np.asarray(
        source_frame.dev_value_from_flux(fluxs), dtype=np.float64
    )
    source_path = _write_v2_csv(
        tmp_path / "row_frame_samples.csv",
        dev_values=source_values,
        dev_unit="A",
        flux_int=source_frame.flux_int,
        flux_period=source_frame.flux_period,
        measurements={"Freq (MHz)": [410.0, 430.0], "T1 (us)": [80.0, 70.0]},
    )

    result = merge_sample_sources(
        target_frame=target,
        sources=(SampleSource(path=source_path, label="row_batch"),),
    )

    np.testing.assert_allclose(result.merged[FLUX_COLUMN], fluxs)
    expected_target_values = np.asarray(
        target.dev_value_from_flux(fluxs), dtype=np.float64
    )
    np.testing.assert_allclose(result.merged[DEV_VALUE_COLUMN], expected_target_values)
    assert list(result.diagnostics["flux_source"]) == ["row-frame", "row-frame"]
    assert result.summary_table.loc[0, "row_frame_flux_rows"] == 2


def test_merge_fallback_frame_provenance(tmp_path: Path) -> None:
    target = _target_frame()
    fallback = FluxFrame(
        params=(3.4, 0.9, 0.5),
        dev_unit="A",
        flux_int=-0.0107,
        flux_period=0.0246,
        label="fallback",
    )
    fluxs = np.array([0.50, 0.55], dtype=np.float64)
    source_values = np.asarray(fallback.dev_value_from_flux(fluxs), dtype=np.float64)
    # Migrated rows: only dev_value / dev_unit, no flux or row frame.
    source_path = _write_v2_csv(
        tmp_path / "migrated_samples.csv",
        dev_values=source_values,
        dev_unit="A",
        measurements={"Freq (MHz)": [410.0, 430.0], "T1 (us)": [80.0, 70.0]},
    )

    result = merge_sample_sources(
        target_frame=target,
        sources=(
            SampleSource(path=source_path, label="migrated", fallback_frame=fallback),
        ),
    )

    np.testing.assert_allclose(result.merged[FLUX_COLUMN], fluxs)
    assert list(result.diagnostics["flux_source"]) == [
        "fallback-frame",
        "fallback-frame",
    ]
    assert result.summary_table.loc[0, "fallback_frame_flux_rows"] == 2


def test_merge_fallback_frame_unit_mismatch_fails(tmp_path: Path) -> None:
    target = _target_frame()
    fallback = FluxFrame(
        params=(3.4, 0.9, 0.5),
        dev_unit="A",
        flux_int=-0.0107,
        flux_period=0.0246,
        label="fallback_A",
    )
    source_path = _write_v2_csv(
        tmp_path / "voltage_migrated.csv",
        dev_values=np.array([0.0, 0.05]),
        dev_unit="V",
    )

    with pytest.raises(SampleTableV2Error, match="unable to resolve flux"):
        merge_sample_sources(
            target_frame=target,
            sources=(SampleSource(path=source_path, fallback_frame=fallback),),
        )


def test_merge_unresolved_rows_fail_with_indexes(tmp_path: Path) -> None:
    target = _target_frame()
    source_path = _write_v2_csv(
        tmp_path / "unresolved_samples.csv",
        dev_values=np.array([1.0, 2.0, 3.0]),
        dev_unit="A",
        flux=np.array([0.5, np.nan, np.nan]),
    )

    with pytest.raises(SampleTableV2Error, match=r"row\(s\) \[1, 2\]"):
        merge_sample_sources(
            target_frame=target,
            sources=(SampleSource(path=source_path),),
        )


def test_merge_rejects_legacy_columns(tmp_path: Path) -> None:
    target = _target_frame()
    legacy_path = _write_v2_csv(
        tmp_path / "legacy.csv",
        dev_values=np.array([1.0, 2.0]),
        dev_unit="A",
        measurements={"calibrated mA": [1.0, 2.0], "Freq (MHz)": [410.0, 430.0]},
    )
    with pytest.raises(SampleTableV2Error, match="calibrated mA"):
        merge_sample_sources(
            target_frame=target,
            sources=(SampleSource(path=legacy_path),),
        )

    flux_alias_path = _write_v2_csv(
        tmp_path / "legacy_flux.csv",
        dev_values=np.array([1.0, 2.0]),
        dev_unit="A",
        measurements={"Flux": [0.5, 0.55]},
    )
    with pytest.raises(SampleTableV2Error, match="Flux"):
        merge_sample_sources(
            target_frame=target,
            sources=(SampleSource(path=flux_alias_path),),
        )


def test_merge_a_and_v_sources_into_one_target_frame(tmp_path: Path) -> None:
    target = _target_frame(dev_unit="A")
    a_frame = FluxFrame(
        params=(3.4, 0.9, 0.5),
        dev_unit="A",
        flux_int=-0.0107,
        flux_period=0.0246,
        label="a_frame",
    )
    v_frame = FluxFrame(
        params=(3.0, 1.0, 0.4),
        dev_unit="V",
        flux_int=0.0,
        flux_period=0.05,
        label="v_frame",
    )
    a_fluxs = np.array([0.50, 0.55], dtype=np.float64)
    v_fluxs = np.array([-0.40, 0.10], dtype=np.float64)
    a_values = np.asarray(a_frame.dev_value_from_flux(a_fluxs), dtype=np.float64)
    v_values = np.asarray(v_frame.dev_value_from_flux(v_fluxs), dtype=np.float64)
    a_path = _write_v2_csv(
        tmp_path / "a_source.csv",
        dev_values=a_values,
        dev_unit="A",
        flux_int=a_frame.flux_int,
        flux_period=a_frame.flux_period,
        measurements={"Freq (MHz)": [410.0, 430.0], "T1 (us)": [80.0, 70.0]},
    )
    v_path = _write_v2_csv(
        tmp_path / "v_source.csv",
        dev_values=v_values,
        dev_unit="V",
        flux_int=v_frame.flux_int,
        flux_period=v_frame.flux_period,
        measurements={"Freq (MHz)": [520.0, 540.0], "T1 (us)": [60.0, 50.0]},
    )

    result = merge_sample_sources(
        target_frame=target,
        sources=(
            SampleSource(path=a_path, label="a_source"),
            SampleSource(path=v_path, label="v_source"),
        ),
    )

    expected_fluxs = np.concatenate([a_fluxs, v_fluxs])
    expected_values = np.asarray(
        target.dev_value_from_flux(expected_fluxs), dtype=np.float64
    )
    np.testing.assert_allclose(result.merged[FLUX_COLUMN], expected_fluxs)
    np.testing.assert_allclose(result.merged[DEV_VALUE_COLUMN], expected_values)
    assert (result.merged[DEV_UNIT_COLUMN] == "A").all()
    np.testing.assert_allclose(result.merged[FLUX_INT_COLUMN], target.flux_int)
    np.testing.assert_allclose(result.merged[FLUX_PERIOD_COLUMN], target.flux_period)
    np.testing.assert_allclose(result.merged["T1 (us)"], [80.0, 70.0, 60.0, 50.0])
    assert list(result.diagnostics["flux_source"]) == [
        "row-frame",
        "row-frame",
        "row-frame",
        "row-frame",
    ]
    assert list(result.diagnostics["source_label"]) == [
        "a_source",
        "a_source",
        "v_source",
        "v_source",
    ]
    # Raw A and V device values are never compared directly.
    assert (result.diagnostics["dev_unit"] == ["A", "A", "V", "V"]).all()
    assert len(result.summary_table) == 2


def test_integer_flux_offset_aligns_equivalent_branches(tmp_path: Path) -> None:
    target = _target_frame()
    zero_path = _write_v2_csv(
        tmp_path / "zero.csv",
        dev_values=np.array([1.0, 2.0]),
        dev_unit="A",
        flux=np.array([0.5, 0.55]),
    )
    minus_half_path = _write_v2_csv(
        tmp_path / "minus_half.csv",
        dev_values=np.array([1.0, 2.0]),
        dev_unit="A",
        flux=np.array([-0.5, -0.45]),
    )
    plus_half_path = _write_v2_csv(
        tmp_path / "plus_half.csv",
        dev_values=np.array([1.0, 2.0]),
        dev_unit="A",
        flux=np.array([0.5, 0.55]),
    )

    result = merge_sample_sources(
        target_frame=target,
        sources=(
            SampleSource(path=zero_path, label="zero"),
            SampleSource(path=minus_half_path, label="plus_one", integer_flux_offset=1),
            SampleSource(
                path=plus_half_path, label="minus_one", integer_flux_offset=-1
            ),
        ),
    )

    merged = result.merged
    # 0.5 (offset 0) and -0.5 (offset +1) land on the same target coordinate.
    np.testing.assert_allclose(merged[FLUX_COLUMN], [0.5, 0.55, 0.5, 0.55, -0.5, -0.45])
    np.testing.assert_allclose(
        merged.loc[0, DEV_VALUE_COLUMN], merged.loc[2, DEV_VALUE_COLUMN]
    )
    expected_for_plus_one = np.asarray(
        target.dev_value_from_flux(np.array([0.5, 0.55])), dtype=np.float64
    )
    np.testing.assert_allclose(merged.loc[2:3, DEV_VALUE_COLUMN], expected_for_plus_one)
    expected_for_minus_one = np.asarray(
        target.dev_value_from_flux(np.array([-0.5, -0.45])), dtype=np.float64
    )
    np.testing.assert_allclose(
        merged.loc[4:5, DEV_VALUE_COLUMN], expected_for_minus_one
    )
    assert list(result.diagnostics["integer_flux_offset"]) == [0, 0, 1, 1, -1, -1]


def test_merge_does_not_guess_integer_branch_without_offset(tmp_path: Path) -> None:
    target = _target_frame()
    minus_half_path = _write_v2_csv(
        tmp_path / "minus_half.csv",
        dev_values=np.array([1.0, 2.0]),
        dev_unit="A",
        flux=np.array([-0.5, -0.45]),
    )

    result = merge_sample_sources(
        target_frame=target,
        sources=(SampleSource(path=minus_half_path, label="no_offset"),),
    )

    # No caller-declared offset: the resolved -0.5 branch is preserved as-is.
    np.testing.assert_allclose(result.merged[FLUX_COLUMN], [-0.5, -0.45])
    assert (result.diagnostics["integer_flux_offset"] == 0).all()


def test_integer_flux_offset_must_be_an_int(tmp_path: Path) -> None:
    source_path = _write_v2_csv(
        tmp_path / "samples.csv",
        dev_values=np.array([1.0]),
        dev_unit="A",
        flux=np.array([0.5]),
    )
    with pytest.raises(ValueError, match="integer_flux_offset must be an int"):
        SampleSource(path=source_path, integer_flux_offset=1.0)  # type: ignore[arg-type]


def test_merge_fits_one_batch_flux_offset_against_target_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target_frame(flux_int=0.0, flux_period=1.0)
    raw_fluxs = np.array([0.45, 0.50, 0.55], dtype=np.float64)
    true_offset = 0.014
    max_abs = 0.05
    source_values = np.asarray(target.dev_value_from_flux(raw_fluxs), dtype=np.float64)
    source_path = _write_v2_csv(
        tmp_path / "samples.csv",
        dev_values=source_values,
        dev_unit="A",
        flux=raw_fluxs,
        measurements={
            "Freq (MHz)": 1000.0 + 200.0 * (raw_fluxs + true_offset),
            "T1 (us)": [10.0, 10.0, 10.0],
        },
    )

    def _linear_f01(
        _params: tuple[float, float, float], fluxs: np.ndarray, *, cutoff: int = 40
    ) -> np.ndarray:
        return 1000.0 + 200.0 * np.asarray(fluxs, dtype=np.float64)

    monkeypatch.setattr(sample_merge, "predict_f01_mhz", _linear_f01)

    result = merge_sample_sources(
        target_frame=target,
        sources=(
            SampleSource(
                path=source_path,
                label="batch",
                fit_batch_flux_offset=True,
                max_abs_batch_flux_offset=max_abs,
            ),
        ),
    )

    fitted = float(result.summary_table.loc[0, "fitted_flux_offset"])
    assert fitted == pytest.approx(true_offset, abs=2e-5)
    assert abs(fitted) <= max_abs
    np.testing.assert_allclose(
        result.merged[FLUX_COLUMN], raw_fluxs + true_offset, atol=2e-5
    )
    assert "residual_before_offset_MHz" in result.diagnostics.columns
    assert "residual_after_merge_MHz" in result.diagnostics.columns
    assert "residual_median_abs_before_offset_MHz" in result.summary_table.columns
    assert "residual_median_abs_after_merge_MHz" in result.summary_table.columns
    assert bool(result.summary_table.loc[0, "batch_fit_success"]) is True


def test_sample_source_rejects_manual_flux_offset(tmp_path: Path) -> None:
    """The unapproved manual offset field is not part of the Interface."""
    with pytest.raises(TypeError):
        SampleSource(
            path=str(tmp_path / "samples.csv"),
            label="manual",
            manual_flux_offset=1.0,  # type: ignore[call-arg]
        )


def test_batch_fit_cannot_shift_flux_across_integer_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The only non-integer post-branch adjustment is the bounded batch fit.

    A source flux of -0.5 with ``integer_flux_offset=0`` must stay near -0.5
    even when a batch fit runs; reaching +0.5 requires the caller-declared
    ``integer_flux_offset=1``.
    """
    target = _target_frame(flux_int=0.0, flux_period=1.0)
    raw_fluxs = np.array([-0.5, -0.45], dtype=np.float64)
    max_abs = 0.03
    source_values = np.asarray(target.dev_value_from_flux(raw_fluxs), dtype=np.float64)
    source_path = _write_v2_csv(
        tmp_path / "samples.csv",
        dev_values=source_values,
        dev_unit="A",
        flux=raw_fluxs,
        # data consistent with flux -0.47 / -0.42: batch fit would like +0.03
        measurements={"Freq (MHz)": 1000.0 + 200.0 * (raw_fluxs + max_abs)},
    )

    def _linear_f01(
        _params: tuple[float, float, float], fluxs: np.ndarray, *, cutoff: int = 40
    ) -> np.ndarray:
        return 1000.0 + 200.0 * np.asarray(fluxs, dtype=np.float64)

    monkeypatch.setattr(sample_merge, "predict_f01_mhz", _linear_f01)

    result = merge_sample_sources(
        target_frame=target,
        sources=(
            SampleSource(
                path=source_path,
                label="branch",
                integer_flux_offset=0,
                fit_batch_flux_offset=True,
                max_abs_batch_flux_offset=max_abs,
            ),
        ),
    )

    fitted = float(result.summary_table.loc[0, "fitted_flux_offset"])
    assert fitted == pytest.approx(max_abs, abs=1e-4)
    merged_flux = result.merged[FLUX_COLUMN].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(merged_flux, raw_fluxs + max_abs, atol=1e-4)
    assert np.all(merged_flux < -0.4), "must not cross to the +0.5 branch"
    assert np.all(np.abs(merged_flux - 0.5) > 0.4), (
        "reaching +0.5 must require integer_flux_offset=1"
    )
    assert "manual_flux_offset" not in result.diagnostics.columns
    assert "manual_flux_offset" not in result.summary_table.columns
    assert "total_flux_offset" not in result.summary_table.columns


def test_write_merged_samples_never_overwrites_a_source_csv(
    tmp_path: Path,
) -> None:
    target = _target_frame()
    source_path = _write_v2_csv(
        tmp_path / "samples.csv",
        dev_values=np.array([1.0, 2.0]),
        dev_unit="A",
        flux=np.array([0.5, 0.55]),
        measurements={"T1 (us)": [80.0, 70.0]},
    )
    result = merge_sample_sources(
        target_frame=target,
        sources=(SampleSource(path=source_path, label="s"),),
    )
    original_bytes = source_path.read_bytes()

    with pytest.raises(ValueError, match="refusing to overwrite source CSV"):
        write_merged_samples(result, source_path)
    assert source_path.read_bytes() == original_bytes

    with pytest.raises(ValueError, match="refusing to overwrite source CSV"):
        write_sample_merge_report(result, source_path)
    assert source_path.read_bytes() == original_bytes

    output_path = tmp_path / "merged" / "samples.csv"
    written = write_merged_samples(result, output_path)
    assert written == output_path
    assert output_path.exists()
    round_trip = pd.read_csv(output_path)
    np.testing.assert_allclose(round_trip[FLUX_COLUMN], [0.5, 0.55])

    report_path = tmp_path / "merged" / "report.csv"
    write_sample_merge_report(result, report_path)
    assert report_path.exists()


def test_write_guards_protect_zero_row_header_only_sources(
    tmp_path: Path,
) -> None:
    """Zero-row sources have no row diagnostics, so per-source paths must
    still protect the original (empty) source CSV from being overwritten.
    """
    target = _target_frame()
    source_path = _write_v2_csv(
        tmp_path / "samples.csv",
        dev_values=np.asarray([], dtype=np.float64),
        dev_unit="A",
        flux=np.asarray([], dtype=np.float64),
    )
    source_bytes = source_path.read_bytes()

    result = merge_sample_sources(
        target_frame=target,
        sources=(SampleSource(path=source_path, label="empty"),),
    )
    assert len(result.merged) == 0
    assert list(result.summary_table["path"]) == [str(source_path.resolve())]

    with pytest.raises(ValueError, match="refusing to overwrite source CSV"):
        write_merged_samples(result, source_path)
    assert source_path.read_bytes() == source_bytes

    with pytest.raises(ValueError, match="refusing to overwrite source CSV"):
        write_sample_merge_report(result, source_path)
    assert source_path.read_bytes() == source_bytes

    # a caller-owned output path is still writable
    out_path = tmp_path / "merged_zero_row" / "samples.csv"
    assert write_merged_samples(result, out_path) == out_path
    assert out_path.exists()
    report_path = tmp_path / "merged_zero_row" / "report.csv"
    write_sample_merge_report(result, report_path)
    assert report_path.exists()


def _write_params(result_dir: Path) -> Path:
    result_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "project": {
            "chip_name": result_dir.name,
            "qubit_name": "Q1",
            "resonator_name": "unknown",
        },
        "fluxdep_fit": {
            "params": {"EJ": 3.5, "EC": 1.0, "EL": 0.6},
            "flux_half": -11.1,
            "flux_int": 1.15,
            "flux_period": 24.5,
            "plot_transitions": {},
        },
    }
    (result_dir / "params.json").write_text(json.dumps(payload), encoding="utf-8")
    return result_dir
