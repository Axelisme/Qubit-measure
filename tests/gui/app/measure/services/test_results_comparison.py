"""Saved result comparisons preserve values and historical quality facts."""

from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.services.results_comparison import build_comparison
from zcu_tools.gui.app.measure.services.results_index import ResultsIndex

from tests.gui.app.measure._comparison_support import saved_run


def test_mixed_tones_keep_names_fit_facts_failures_and_input_order(
    tmp_path: Path,
) -> None:
    mux = saved_run(
        tmp_path,
        "mux",
        "mux/t2ramsey",
        {
            "tones": [
                {
                    "name": "Q5",
                    "t2": 11.5,
                    "fit_quality": {
                        "decay": {"r2": 0.03, "invalid": []},
                        "fringe": {
                            "r2": 0.9,
                            "invalid": [{"path": "amplitude", "reason": "singular"}],
                        },
                    },
                },
                {"label": "Q9", "t2": 8, "fit_quality": {"r2": 0.7, "invalid": False}},
                {"index": 7, "failure": "fit did not converge"},
            ]
        },
    )
    resonator = saved_run(
        tmp_path,
        "readout",
        "mux/onetone",
        {
            "resonators": [
                {"label": "R2", "freq": 6.125, "kappa": 0.002},
                {"name": "R8", "freq": 7.25},
            ]
        },
    )
    unknown = saved_run(
        tmp_path, "unknown", "custom/t1ish", {"t1": 999, "failure": "saved failure"}
    )
    table = build_comparison(
        ResultsIndex(tmp_path, (mux, resonator, unknown)),
        (unknown.key, resonator.key, mux.key),
    )
    assert tuple(row.run_key for row in table.rows) == (
        unknown.key,
        resonator.key,
        mux.key,
    )
    assert table.columns == (
        "entry",
        "point",
        "experiment",
        "time",
        "t2",
        "t2_err",
        "detune",
        "freq",
        "fwhm",
        "kappa",
        "depth",
        "r2",
        "invalid",
        "failure",
        "issues",
    )
    cells = dict(zip(table.columns, table.rows[2].cells, strict=True))
    assert cells["t2"] == "Q5: 11.5\nQ9: 8\n7: —"
    assert cells["r2"] == "Q5 / decay: 0.03\nQ5 / fringe: 0.9\nQ9: 0.7\n7: —"
    assert (
        cells["invalid"]
        == 'Q5 / decay: []\nQ5 / fringe: [{"path": "amplitude", "reason": "singular"}]\nQ9: false\n7: —'
    )
    assert cells["failure"] == "Q5: —\nQ9: —\n7: fit did not converge"
    assert table.rows[1].cells[7] == "R2: 6.125\nR8: 7.25"
    assert table.rows[0].cells[-2] == "saved failure"


def test_registered_spectrum_name_selects_saved_frequency_fields(
    tmp_path: Path,
) -> None:
    run = saved_run(
        tmp_path,
        "spectrum",
        "twotone/freq",
        {"freq": 5123.75, "freq_err": 0.1, "fwhm": 2, "unrelated": 4},
    )
    table = build_comparison(ResultsIndex(tmp_path, (run,)), (run.key,))
    assert table.columns[4:7] == ("freq", "freq_err", "fwhm")
    assert table.rows[0].cells[4:7] == ("5123.75", "0.1", "2")


def test_direct_quality_diagnostics_keep_tone_label(tmp_path: Path) -> None:
    run = saved_run(
        tmp_path,
        "direct",
        "mux/t1",
        {
            "tones": [
                {
                    "name": "Q4",
                    "t1": 20,
                    "fit_quality": {
                        "r2": 0.02,
                        "invalid": [{"path": "t1", "reason": "saved diagnostic"}],
                    },
                }
            ]
        },
    )
    table = build_comparison(ResultsIndex(tmp_path, (run,)), (run.key,))
    assert (
        table.rows[0].cells[-3] == 'Q4: [{"path": "t1", "reason": "saved diagnostic"}]'
    )


def test_unknown_keys_raise_and_broken_files_remain_visible(tmp_path: Path) -> None:
    run = saved_run(tmp_path, "broken", None, {})
    run.analysis_paths[0].write_text("broken", encoding="utf-8")
    index = ResultsIndex(tmp_path, (run,))
    with pytest.raises(KeyError):
        build_comparison(index, ("missing",))
    table = build_comparison(index, (run.key,))
    assert table.rows[0].cells[2:4] == ("—", "—")
    assert "analysis.json" in table.rows[0].cells[-1]


def test_flat_saved_summary_uses_fixed_fields_and_missing_values(
    tmp_path: Path,
) -> None:
    run = saved_run(
        tmp_path,
        "flat",
        "twotone/time_domain/t1",
        {"t1": 21.25, "t1_err": 0, "extra": 999, "r2": 0.01},
    )
    table = build_comparison(ResultsIndex(tmp_path, (run,)), (run.key,))
    assert table.columns == (
        "entry",
        "point",
        "experiment",
        "time",
        "t1",
        "t1_err",
        "t1b",
        "r2",
        "invalid",
        "failure",
        "issues",
    )
    assert [(row.run_key, row.cells) for row in table.rows] == [
        (
            run.key,
            (
                "sample",
                "flat",
                "twotone/time_domain/t1",
                "—",
                "21.25",
                "0",
                "—",
                "0.01",
                "—",
                "—",
                "—",
            ),
        )
    ]
