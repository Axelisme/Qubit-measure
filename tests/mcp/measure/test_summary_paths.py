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
