"""Captured native analysis values selected by typed original-position paths."""

from dataclasses import replace
from typing import Any

import pytest
from zcu_tools.mcp.measure.execution_reply import (
    SummaryEstimate,
    SummaryParameter,
    project_execution,
)

from ._recipe_execution_support import definition


def unused_recipe(_session):
    yield from ()


def analysis_snapshot() -> dict[str, Any]:
    return {
        "execution": "a",
        "stage": "primary",
        "tab": "t",
        "op": 1,
        "status": "finished",
        "phase": "finished",
        "cancel_requested": False,
        "writeback": None,
        "error": None,
        "operation_outcome": {"status": "finished"},
        "save_status": "not_started",
        "result": {
            "summary": {
                "tones": [{"fit": 1.0}, {"fit": 2.5, "err": None}],
                "0": {"fit": 7},
            },
            "invalid": [{"path": ["tones", 1, "err"], "reason": "non_finite"}],
        },
    }


@pytest.mark.parametrize("declaration", [SummaryParameter, SummaryEstimate])
def test_summary_declarations_reject_string_path_carriers(declaration):
    with pytest.raises(ValueError, match="typed path"):
        declaration("name", "leaf")


def test_summary_looks_up_indexed_and_literal_numeric_name_without_reordering():
    recipe = replace(
        definition(unused_recipe),
        summary_estimates=(
            SummaryEstimate("Q2", ("tones", 1, "fit"), ("tones", 1, "err"), "us"),
            SummaryEstimate("literal", ("0", "fit")),
            SummaryEstimate("missing", ("tones", 2, "fit")),
            SummaryEstimate("string_is_not_index", ("tones", "1", "fit")),
        ),
    )
    snapshot = analysis_snapshot()
    result = project_execution(snapshot, recipes=(recipe,))
    pane = result["analysis"]["primary"]
    assert pane["estimates"] == {
        "Q2": {"value": 2.5, "stderr": None, "unit": "us"},
        "literal": {"value": 7, "stderr": None, "unit": None},
    }
    assert pane["summary"] == snapshot["result"]["summary"]
    assert pane["invalid"] == snapshot["result"]["invalid"]
    pane["summary"]["tones"][1]["fit"] = 999
    assert (
        project_execution(snapshot, recipes=(recipe,))["analysis"]["primary"][
            "estimates"
        ]["Q2"]["value"]
        == 2.5
    )


def test_analysis_only_expands_rows_and_checks_present_path_ambiguity():
    from zcu_tools.mcp.measure import execution_reply as reporting

    assert hasattr(reporting, "SummaryEstimateRows"), "Missing native-row estimate declaration"
    rows = reporting.SummaryEstimateRows(
        "tones", ("tones",),
        (SummaryEstimate("fit", ("fit",), element_key=("element",)),),
    )
    first = replace(definition(unused_recipe), summary_estimates=(rows,))
    snapshot = analysis_snapshot()
    snapshot["result"]["summary"]["tones"] = [
        {"fit": 4002, "position": 2, "name": "Q3", "element": {"index": 2, "name": "Q3"}},
        {"fit": 4000, "position": 0, "name": "Q1"},
        {"position": 1, "name": "Q2"},
    ]
    projected = project_execution(snapshot, recipes=(first, first))
    assert projected["analysis"]["primary"]["estimates"] == {
        "tones/0/fit": {"value": 4002, "stderr": None, "unit": None, "element": {"index": 2, "name": "Q3"}},
        "tones/1/fit": {"value": 4000, "stderr": None, "unit": None},
    }
    conflict = replace(first, name="other", summary_estimates=(
        SummaryEstimate("different", ("tones", 0, "fit"), element_key=("tones", 0, "element")),
    ))
    with pytest.raises(ValueError, match="Ambiguous analysis estimate"):
        project_execution(snapshot, recipes=(first, conflict))
    snapshot["result"]["summary"] = {}
    assert project_execution(snapshot, recipes=(first, conflict))["analysis"]["primary"]["estimates"] == {}


def test_analysis_only_conflicts_are_checked_at_present_typed_path():
    first = replace(
        definition(unused_recipe),
        summary_estimates=(SummaryEstimate("Q2", ("tones", 1, "fit")),),
    )
    conflict = replace(
        first,
        name="other",
        summary_estimates=(SummaryEstimate("other", ("tones", 1, "fit")),),
    )
    with pytest.raises(ValueError, match="Ambiguous analysis estimate"):
        project_execution(analysis_snapshot(), recipes=(first, conflict))
