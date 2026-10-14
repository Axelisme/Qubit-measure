"""v2 SampleTable contract tests for the design-search real-sample overlay.

``add_real_sample`` must select the measured row by normalized resolved flux
(never by direct cross-unit ``dev_value`` comparison). These cases check that
unresolved rows and legacy coordinate columns report public validation errors.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from plotly.graph_objects import Figure
from zcu_tools.notebook.analysis.design import search as S
from zcu_tools.resources.qubit_params import (
    DispersiveFit,
    FluxDepFit,
    ParamsProject,
    QubitParams,
)
from zcu_tools.resources.sample_table import (
    JsonlSampleTable,
    SampleFluxFrame,
    SampleTableV2Error,
)

_NOISE_CHANNELS: list[tuple[str, dict[str, object]]] = []


@pytest.fixture(scope="module", autouse=True)
def guard_shared_state() -> Iterator[None]:
    """Name this module if its tests leak changes to shared recording state."""
    noise_channels = deepcopy(_NOISE_CHANNELS)
    yield
    assert noise_channels == _NOISE_CHANNELS, f"{__name__} leaked noise channels"


@pytest.fixture
def result_dir(tmp_path) -> str:
    """Minimal parameter directory for public validation errors."""
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

    return str(result)


@pytest.fixture
def samples(tmp_path: Path) -> JsonlSampleTable:
    return JsonlSampleTable(tmp_path / "caller-samples.jsonl")


def _write_sample_jsonl(
    samples: JsonlSampleTable, rows: list[dict[str, object]]
) -> None:
    _write_sample_frame(samples, pd.DataFrame(rows))


def _write_sample_frame(samples: JsonlSampleTable, frame: pd.DataFrame) -> None:

    frame = frame.copy()
    frame["run_id"] = [f"run-{index}" for index in range(len(frame))]
    frame["point"] = None
    frame["at"] = "2026-10-10T00:00:00Z"
    frame["source"] = "manual"
    if "dev_value" not in frame:
        frame["dev_value"] = 0.0
        frame["dev_unit"] = "A"
    path = samples.path
    if frame.empty:
        with open(path, "w", encoding="utf-8"):
            pass
    else:
        frame.to_json(path, orient="records", lines=True)


def test_unresolved_rows_report_indexes(
    result_dir: str, samples: JsonlSampleTable
) -> None:
    _write_sample_jsonl(
        samples,
        [
            {"dev_value": 0.3, "dev_unit": "A", "flux": 0.3, "t1_us": 11.0},
            {"dev_value": 0.6, "dev_unit": "A", "t1_us": 22.0},  # no frame
        ],
    )
    fig = _new_fig()

    with pytest.raises(SampleTableV2Error, match=r"row\(s\) \[1\]"):
        S.add_real_sample(fig, result_dir, _NOISE_CHANNELS, Temp=0.05, samples=samples)


def test_fallback_frame_unit_mismatch_is_unresolved(
    result_dir: str, samples: JsonlSampleTable
) -> None:
    _write_sample_jsonl(
        samples,
        [{"dev_value": 0.5, "dev_unit": "A", "t1_us": 11.0}],
    )
    fig = _new_fig()

    with pytest.raises(SampleTableV2Error, match=r"row\(s\) \[0\]"):
        S.add_real_sample(
            fig,
            result_dir,
            _NOISE_CHANNELS,
            Temp=0.05,
            samples=samples,
            fallback_frame=SampleFluxFrame("V", flux_int=0.0, flux_period=1.0),
        )


def test_legacy_coordinate_column_is_rejected(
    result_dir: str, samples: JsonlSampleTable
) -> None:
    _write_sample_jsonl(
        samples,
        [
            {"calibrated mA": 0.3, "freq_mhz": 4000.0, "t1_us": 11.0},
        ],
    )
    fig = _new_fig()

    with pytest.raises(SampleTableV2Error, match="calibrated mA"):
        S.add_real_sample(fig, result_dir, _NOISE_CHANNELS, Temp=0.05, samples=samples)


def test_empty_table_is_rejected(result_dir: str, samples: JsonlSampleTable) -> None:
    # Empty JSONL tables report a public validation error.
    _write_sample_frame(
        samples, pd.DataFrame(columns=["dev_value", "dev_unit", "t1_us"])
    )
    fig = _new_fig()

    with pytest.raises(SampleTableV2Error, match="empty"):
        S.add_real_sample(fig, result_dir, _NOISE_CHANNELS, Temp=0.05, samples=samples)


def _new_fig() -> Figure:
    return Figure()
