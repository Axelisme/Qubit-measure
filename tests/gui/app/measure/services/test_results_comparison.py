"""Saved summaries project as neutral JSON, independent of experiment names."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.services.results_comparison import build_comparison
from zcu_tools.gui.app.measure.services.results_index import ResultsIndex

from tests.gui.app.measure._comparison_support import saved_run


@pytest.mark.parametrize("experiment", ["retired/custom", "mux/t2ramsey", "t1"])
def test_summary_paths_union_follows_pin_and_json_order(
    tmp_path: Path, experiment: str
) -> None:
    first = saved_run(
        tmp_path,
        "first",
        experiment,
        {"unknown": 12.5, "fit_quality": {"nested": {"r2": 0.03}}, "zero": 0},
    )
    second = saved_run(
        tmp_path, "second", "arbitrary/name", {"zero": False, "later": "saved"}
    )
    index = ResultsIndex(tmp_path, (first, second))
    table = build_comparison(index, (first.key, second.key))
    assert table.columns == (
        "entry",
        "point",
        "experiment",
        "time",
        "unknown",
        "fit_quality.nested.r2",
        "zero",
        "later",
    )
    assert table.rows[0].cells == (
        "sample",
        "first",
        experiment,
        "—",
        "12.5",
        "0.03",
        "0",
        "—",
    )
    assert table.rows[1].cells == (
        "sample",
        "second",
        "arbitrary/name",
        "—",
        "—",
        "—",
        "false",
        "saved",
    )
    reversed_table = build_comparison(index, (second.key, first.key))
    assert reversed_table.columns[4:] == (
        "zero",
        "later",
        "unknown",
        "fit_quality.nested.r2",
    )
    assert tuple(row.run_key for row in reversed_table.rows) == (second.key, first.key)


def test_object_lists_union_leaf_paths_and_use_indices_not_labels(
    tmp_path: Path,
) -> None:
    run = saved_run(
        tmp_path,
        "lists",
        "unknown/object-list",
        {
            "items": [
                {"name": "Q5", "value": 2.5, "nested": {"enabled": True}},
                {"label": "Q9", "nested": {"extra": "saved"}},
                {},
            ],
            "scalars": [1, "two", None, False],
            "mixed": [{"value": 3}, 4],
            "empty_list": [],
            "empty_object": {},
            "null_value": None,
        },
    )
    table = build_comparison(ResultsIndex(tmp_path, (run,)), (run.key,))
    assert table.columns[4:] == (
        "items.name",
        "items.value",
        "items.nested.enabled",
        "items.label",
        "items.nested.extra",
        "scalars",
        "mixed",
        "empty_list",
        "null_value",
    )
    assert table.rows[0].cells[4:] == (
        "0: Q5\n1: —\n2: —",
        "0: 2.5\n1: —\n2: —",
        "0: true\n1: —\n2: —",
        "0: —\n1: Q9\n2: —",
        "0: —\n1: saved\n2: —",
        '[1, "two", null, false]',
        '[{"value": 3}, 4]',
        "[]",
        "null",
    )


def test_analysis_members_merge_without_parameters_and_isolate_bad_json(
    tmp_path: Path,
) -> None:
    run = saved_run(tmp_path, "members", None, {"entry": "saved entry", "value": 1})
    (first,) = run.analysis_paths
    first.write_text(
        json.dumps(
            {
                "summary": {"entry": "saved entry", "value": 1},
                "parameters": {"hidden_parameter": 999},
            }
        ),
        encoding="utf-8",
    )
    second = first.with_name("second.json")
    second.write_text(
        json.dumps({"summary": {"value": 2, "later": None}}), encoding="utf-8"
    )
    broken = first.with_name("broken.json")
    broken.write_text("broken JSON", encoding="utf-8")
    run = replace(run, analysis_paths=(first, second, broken))
    index = ResultsIndex(tmp_path, (run,))
    with pytest.raises(KeyError):
        build_comparison(index, ("missing",))
    table = build_comparison(index, (run.key,))
    assert table.columns == (
        "entry",
        "point",
        "experiment",
        "time",
        "entry",
        "value",
        "later",
    )
    assert table.rows[0].cells == (
        "sample",
        "members",
        "—",
        "—",
        "analysis.json: saved entry",
        "analysis.json: 1\nsecond.json: 2",
        "second.json: null",
    )
    assert build_comparison(index, ()).columns == (
        "entry",
        "point",
        "experiment",
        "time",
    )
    assert build_comparison(index, ()).rows == ()
