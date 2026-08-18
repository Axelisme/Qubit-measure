"""v2 SampleTable contract tests for the design-search real-sample overlay.

``add_real_sample`` must select the measured row by normalized resolved flux
(never by direct cross-unit ``dev_value`` comparison): explicit flux, row-frame
derivation and the caller-declared fallback frame all select the row whose
resolved flux is nearest the operating point; unresolved rows and legacy
coordinate columns fail before any physics or comparison runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Scatter
from zcu_tools.meta_tool import (
    DispersiveFit,
    FluxDepFit,
    ParamsProject,
    QubitParams,
    SampleFluxFrame,
    SampleTableV2Error,
)
from zcu_tools.notebook.analysis.design import search as S

_NOISE_CHANNELS: list[tuple[str, dict[str, object]]] = []

# Physics-call counter shared with the fake implementations; reset per test.
_PHYSICS_CALLS = {"physics": 0}


def _actual_t1_trace(fig) -> Scatter:
    traces = [trace for trace in fig.data if trace.mode == "markers+text"]
    assert len(traces) == 1
    return traces[0]


@pytest.fixture
def result_dir(tmp_path, monkeypatch) -> str:
    """Minimal result dir (params.json + no sample.csv yet) with fake physics."""
    result = tmp_path / "q1"
    result.mkdir()
    params_file = QubitParams(result / "params.json")
    params_file.ensure_project(ParamsProject(chip_name="chip", qub_name="q1"))
    params_file.set_fluxdep_fit(
        FluxDepFit(
            EJ=4.0,
            EC=1.0,
            EL=1.0,
            flux_half=0.5,
            flux_int=0.0,
            flux_period=1.0,
        )
    )
    params_file.set_dispersive_fit(DispersiveFit(g=0.1, bare_rf=7.0))

    calls = _PHYSICS_CALLS
    calls["physics"] = 0

    def _fake_calc_ge_snr(*_args, **_kwargs):
        calls["physics"] += 1
        return None, np.array([1.0, 2.0, 3.0])

    import scqubits.core.fluxonium as fluxonium_mod

    class FakeFluxonium:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def t1_effective(self, *_args, **_kwargs) -> float:
            calls["physics"] += 1
            return 500.0  # ns -> predict_t1 = 0.5 us

    monkeypatch.setattr(S, "calc_ge_snr", _fake_calc_ge_snr)
    monkeypatch.setattr(fluxonium_mod, "Fluxonium", FakeFluxonium)
    return str(result)


def _write_sample_csv(result_dir: str, rows: list[dict[str, object]]) -> None:
    import os

    pd.DataFrame(rows).to_csv(os.path.join(result_dir, "sample.csv"), index=False)


def _write_sample_frame(result_dir: str, frame: pd.DataFrame) -> None:
    import os

    frame.to_csv(os.path.join(result_dir, "sample.csv"), index=False)


def test_selects_row_by_explicit_flux(result_dir: str) -> None:
    _write_sample_csv(
        result_dir,
        [
            {"dev_value": 0.3, "dev_unit": "A", "flux": 0.3, "T1 (us)": 11.0},
            {"dev_value": 0.5, "dev_unit": "A", "flux": 0.5, "T1 (us)": 22.0},
            {"dev_value": 0.7, "dev_unit": "A", "flux": 0.7, "T1 (us)": 33.0},
        ],
    )
    fig = _new_fig()
    S.add_real_sample(fig, result_dir, _NOISE_CHANNELS, Temp=0.05)

    trace = _actual_t1_trace(fig)
    assert float(trace.y[0]) == 22.0
    # predicted point comes from the fake model, not from the table
    predicted = [t for t in fig.data if t.mode == "markers"]
    assert len(predicted) == 1
    assert float(predicted[0].y[0]) == 0.5


def test_selects_row_by_row_frame_derived_flux(result_dir: str) -> None:
    _write_sample_csv(
        result_dir,
        [
            {
                "dev_value": 0.0,
                "dev_unit": "A",
                "flux_int": 0.0,
                "flux_period": 2.0,
                "T1 (us)": 11.0,
            },  # flux 0.0
            {
                "dev_value": 1.0,
                "dev_unit": "A",
                "flux_int": 0.0,
                "flux_period": 2.0,
                "T1 (us)": 22.0,
            },  # flux 0.5
            {
                "dev_value": 1.4,
                "dev_unit": "A",
                "flux_int": 0.0,
                "flux_period": 2.0,
                "T1 (us)": 33.0,
            },  # flux 0.7
        ],
    )
    fig = _new_fig()
    S.add_real_sample(fig, result_dir, _NOISE_CHANNELS, Temp=0.05)

    assert float(_actual_t1_trace(fig).y[0]) == 22.0


def test_selects_row_by_declared_fallback_frame(result_dir: str) -> None:
    _write_sample_csv(
        result_dir,
        [
            {"dev_value": 0.3, "dev_unit": "A", "T1 (us)": 11.0},
            {"dev_value": 0.5, "dev_unit": "A", "T1 (us)": 22.0},
        ],
    )
    fig = _new_fig()
    S.add_real_sample(
        fig,
        result_dir,
        _NOISE_CHANNELS,
        Temp=0.05,
        fallback_frame=SampleFluxFrame("A", flux_int=0.0, flux_period=1.0),
    )

    assert float(_actual_t1_trace(fig).y[0]) == 22.0


def test_cross_unit_rows_compare_in_flux_domain_only(result_dir: str) -> None:
    # The V row is flux-nearest (0.5) although its dev_value (3.0 V) is far from
    # the A row values; direct dev_value comparison across units would pick row0.
    _write_sample_csv(
        result_dir,
        [
            {
                "dev_value": 0.49,
                "dev_unit": "A",
                "flux_int": 0.0,
                "flux_period": 1.0,
                "T1 (us)": 11.0,
            },  # flux 0.49
            {
                "dev_value": 3.0,
                "dev_unit": "V",
                "flux_int": 2.0,
                "flux_period": 2.0,
                "T1 (us)": 22.0,
            },  # flux 0.5
        ],
    )
    fig = _new_fig()
    S.add_real_sample(fig, result_dir, _NOISE_CHANNELS, Temp=0.05)

    assert float(_actual_t1_trace(fig).y[0]) == 22.0


def test_unresolved_rows_fail_with_indexes_before_physics(result_dir: str) -> None:
    _write_sample_csv(
        result_dir,
        [
            {"dev_value": 0.3, "dev_unit": "A", "flux": 0.3, "T1 (us)": 11.0},
            {"dev_value": 0.6, "dev_unit": "A", "T1 (us)": 22.0},  # no frame
        ],
    )
    fig = _new_fig()

    with pytest.raises(SampleTableV2Error, match=r"row\(s\) \[1\]"):
        S.add_real_sample(fig, result_dir, _NOISE_CHANNELS, Temp=0.05)

    assert _PHYSICS_CALLS["physics"] == 0


def test_fallback_frame_unit_mismatch_is_unresolved(result_dir: str) -> None:
    _write_sample_csv(
        result_dir,
        [{"dev_value": 0.5, "dev_unit": "A", "T1 (us)": 11.0}],
    )
    fig = _new_fig()

    with pytest.raises(SampleTableV2Error, match=r"row\(s\) \[0\]"):
        S.add_real_sample(
            fig,
            result_dir,
            _NOISE_CHANNELS,
            Temp=0.05,
            fallback_frame=SampleFluxFrame("V", flux_int=0.0, flux_period=1.0),
        )


def test_legacy_coordinate_column_fails_before_physics(result_dir: str) -> None:
    _write_sample_csv(
        result_dir,
        [
            {"calibrated mA": 0.3, "Freq (MHz)": 4000.0, "T1 (us)": 11.0},
        ],
    )
    fig = _new_fig()

    with pytest.raises(SampleTableV2Error, match="calibrated mA"):
        S.add_real_sample(fig, result_dir, _NOISE_CHANNELS, Temp=0.05)

    assert _PHYSICS_CALLS["physics"] == 0


def test_empty_table_fails_before_physics(result_dir: str) -> None:
    # Header-only v2 table (zero rows): validation must fail before physics.
    _write_sample_frame(
        result_dir, pd.DataFrame(columns=["dev_value", "dev_unit", "T1 (us)"])
    )
    fig = _new_fig()

    with pytest.raises(SampleTableV2Error, match="empty"):
        S.add_real_sample(fig, result_dir, _NOISE_CHANNELS, Temp=0.05)

    assert _PHYSICS_CALLS["physics"] == 0


def _new_fig():
    import plotly.graph_objects as go

    return go.Figure()
