"""B1: tab.get_cfg_summary blanks a sweep node's derived ``step`` when an edge
is an unresolved EvalValue (shown as an expr string), so the reported step never
contradicts the start/stop span the user actually sees. A numeric-edge sweep
keeps its correct derived step.

These exercise the projection-layer post-pass (``_h_tab_get_cfg_summary`` ->
``_null_step_on_unresolved_sweeps``) directly via a minimal controller double;
session_codec (the persistence SSOT) is intentionally untouched."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    EvalValue,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.app.main.services.remote.dispatch import _h_tab_get_cfg_summary


def _adapter_with_schema(schema: CfgSchema) -> MagicMock:
    """A controller double exposing only what _h_tab_get_cfg_summary touches."""
    adapter = MagicMock()
    adapter.ctrl.has_tab.return_value = True
    adapter.ctrl.get_tab_cfg_schema.return_value = schema
    return adapter


def _sweep_schema(
    start: float | EvalValue, stop: float | EvalValue, step: float
) -> CfgSchema:
    spec = CfgSectionSpec(fields={"sweep": SweepSpec(label="Sweep")})
    value = CfgSectionValue(
        fields={"sweep": SweepValue(start=start, stop=stop, expts=11, step=step)}
    )
    return CfgSchema(spec=spec, value=value)


def test_eval_edge_sweep_reports_null_step():
    # An EvalValue edge surfaces as an expr string in the summary; its stored step
    # is stale relative to that expression → reported as None.
    schema = _sweep_schema(
        start=EvalValue(expr="r_f - 10", resolved=5990.0, error=None),
        stop=EvalValue(expr="r_f + 10", resolved=6010.0, error=None),
        step=2.0,
    )
    summary: Any = _h_tab_get_cfg_summary(
        _adapter_with_schema(schema), {"tab_id": "t1"}
    )["summary"]
    sweep = summary["sweep"]
    assert sweep["start"] == "r_f - 10"
    assert sweep["stop"] == "r_f + 10"
    assert sweep["step"] is None


def test_numeric_edge_sweep_keeps_derived_step():
    # Both edges are concrete numbers → the derived step is correct, keep it.
    schema = _sweep_schema(start=0.0, stop=10.0, step=1.0)
    summary: Any = _h_tab_get_cfg_summary(
        _adapter_with_schema(schema), {"tab_id": "t1"}
    )["summary"]
    sweep = summary["sweep"]
    assert sweep["start"] == 0.0
    assert sweep["stop"] == 10.0
    assert sweep["step"] == 1.0


def test_one_eval_edge_is_enough_to_null_step():
    # A single unresolved edge already makes the stored step inconsistent.
    schema = _sweep_schema(
        start=0.0,
        stop=EvalValue(expr="r_f", resolved=6000.0, error=None),
        step=1.0,
    )
    summary: Any = _h_tab_get_cfg_summary(
        _adapter_with_schema(schema), {"tab_id": "t1"}
    )["summary"]
    sweep = summary["sweep"]
    assert sweep["start"] == 0.0
    assert sweep["stop"] == "r_f"
    assert sweep["step"] is None
