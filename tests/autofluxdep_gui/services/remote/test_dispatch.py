"""Tests for the autofluxdep remote dispatch handlers + method-spec validation.

The remote surface is READ-ONLY: the agent observes a GUI the user drives. So
every handler here is a pure query — there is no add-node / set-flux / run / stop
RPC. The tests therefore build the state directly through the ``Controller`` (the
GUI's own command API, exercised by the controller/UI tests) and assert that the
read handlers report it correctly.

Qt-free at the handler level: the handlers only touch ``adapter.ctrl`` (a real
``Controller``), so a tiny stub adapter exercises the whole RPC handler surface
without the socket server. The autouse ``qapp`` fixture (conftest) provides the
QApplication the Controller's session services need at construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.services.remote.dispatch import (
    _HANDLERS,
    METHOD_REGISTRY,
)
from zcu_tools.gui.app.autofluxdep.services.remote.method_specs import METHOD_SPECS
from zcu_tools.gui.app.autofluxdep.state import ProjectInfo
from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.gui.remote.param_spec import validate_params


class _StubAdapter:
    """Minimal stand-in: handlers reach the façade via ``adapter.ctrl`` only."""

    def __init__(self, ctrl: Controller) -> None:
        self.ctrl = ctrl


def _adapter(project: ProjectInfo | None = None) -> _StubAdapter:
    return _StubAdapter(build_core(project=project))


def _call(adapter: _StubAdapter, method: str, raw_params: dict) -> dict:
    """Validate params against the spec (as the service does) then dispatch."""
    spec = METHOD_REGISTRY[method]
    params = validate_params(spec.params, raw_params) if spec.params else raw_params
    return dict(spec.handler(adapter, params))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registry coherence + read-only guard
# ---------------------------------------------------------------------------


def test_registry_specs_and_handlers_match():
    assert set(_HANDLERS) == set(METHOD_SPECS)
    assert set(METHOD_REGISTRY) == set(METHOD_SPECS)


def test_registry_is_read_only():
    # Guard against re-introducing a mutating RPC: only these read methods exist.
    # No add-node / set-flux / run / stop verb may appear here.
    assert set(METHOD_SPECS) == {
        "project.info",
        "workflow.list",
        "node.cfg",
        "result.summary",
        "resources.versions",
        "state.check",
    }


# ---------------------------------------------------------------------------
# project.info
# ---------------------------------------------------------------------------


def test_project_info_null_when_no_project():
    info = _call(_adapter(), "project.info", {})
    assert info == {
        "chip_name": None,
        "qub_name": None,
        "result_dir": None,
        "params_path": None,
    }


def test_project_info_reports_set_project():
    project = ProjectInfo(
        chip_name="Q5_2D",
        qub_name="Q1",
        result_dir="result/Q5_2D/Q1",
        params_path="result/Q5_2D/Q1/params.json",
    )
    info = _call(_adapter(project), "project.info", {})
    assert info["chip_name"] == "Q5_2D"
    assert info["qub_name"] == "Q1"
    assert info["result_dir"] == "result/Q5_2D/Q1"
    assert info["params_path"] == "result/Q5_2D/Q1/params.json"


# ---------------------------------------------------------------------------
# state.check
# ---------------------------------------------------------------------------


def test_state_check_empty():
    check = _call(_adapter(), "state.check", {})
    assert check == {
        "has_project": False,
        "has_soc": False,
        "node_count": 0,
        "flux_count": 0,
        "has_flux_device": False,
        "is_running": False,
        "has_results": False,
        "has_loaded_predictor": False,
        "has_run_predictor": False,
    }


def test_state_check_reflects_workflow_and_project():
    project = ProjectInfo(
        chip_name="Q5_2D", qub_name="Q1", result_dir="r", params_path="p"
    )
    adapter = _adapter(project)
    adapter.ctrl.add_node_by_type("qubit_freq")
    adapter.ctrl.add_node_by_type("t1")
    adapter.ctrl.set_flux_values([0.0, 0.1, 0.2])
    adapter.ctrl.set_flux_device("fake_flux")

    check = _call(adapter, "state.check", {})
    assert check["has_project"] is True
    assert check["node_count"] == 2
    assert check["flux_count"] == 3
    assert check["has_flux_device"] is True
    assert check["is_running"] is False
    assert check["has_results"] is False


# ---------------------------------------------------------------------------
# workflow.list
# ---------------------------------------------------------------------------


def test_workflow_list_reports_placed_nodes():
    adapter = _adapter()
    adapter.ctrl.add_node_by_type("qubit_freq")
    adapter.ctrl.add_node_by_type("lenrabi")

    listed = _call(adapter, "workflow.list", {})["nodes"]
    assert [n["name"] for n in listed] == ["qubit_freq", "lenrabi"]
    qf = listed[0]
    assert qf["type"] == "qubit_freq"
    assert "qubit_freq" in qf["provides"]
    assert qf["has_result"] is False
    # lenrabi declares a dependency on qubit_freq's info key
    assert "qubit_freq" in listed[1]["requires"]


def test_workflow_list_excludes_predictor_service():
    # The predictor service is prepended only at run time; it must never be a row.
    adapter = _adapter()
    adapter.ctrl.add_node_by_type("qubit_freq")
    listed = _call(adapter, "workflow.list", {})["nodes"]
    assert all(n["type"] != "predictor" for n in listed)
    assert len(listed) == 1


def test_workflow_list_has_result_flips_after_allocation():
    adapter = _adapter()
    adapter.ctrl.add_node_by_type("qubit_freq")
    adapter.ctrl.set_flux_values([0.0, 0.1])
    adapter.ctrl.prepare_run_results()
    listed = _call(adapter, "workflow.list", {})["nodes"]
    assert listed[0]["has_result"] is True


# ---------------------------------------------------------------------------
# node.cfg
# ---------------------------------------------------------------------------


def test_node_cfg_reports_knobs():
    adapter = _adapter()
    node = adapter.ctrl.add_node_by_type("qubit_freq")
    adapter.ctrl.set_node_params(0, {"reps": 512})

    cfg = _call(adapter, "node.cfg", {"name": node.name})
    assert cfg["name"] == node.name
    assert cfg["type"] == "qubit_freq"
    knobs = cfg["knobs"]
    # the edited scalar is reflected
    assert knobs["reps"] == 512
    # a sweep knob serialises to {start, stop, expts}, not a SweepCfg
    assert set(knobs["detune_sweep"]) == {"start", "stop", "expts"}


def test_node_cfg_unknown_name_fast_fails():
    adapter = _adapter()
    adapter.ctrl.add_node_by_type("qubit_freq")
    with pytest.raises(KeyError):
        _call(adapter, "node.cfg", {"name": "does_not_exist"})


def test_node_cfg_missing_name_rejected_by_param_validation():
    adapter = _adapter()
    with pytest.raises(RemoteError):
        _call(adapter, "node.cfg", {})


# ---------------------------------------------------------------------------
# result.summary
# ---------------------------------------------------------------------------


def test_result_summary_empty_when_no_run():
    adapter = _adapter()
    adapter.ctrl.add_node_by_type("qubit_freq")
    summary = _call(adapter, "result.summary", {})
    assert summary == {"results": []}


def test_result_summary_reports_progress():
    adapter = _adapter()
    node = adapter.ctrl.add_node_by_type("qubit_freq")
    adapter.ctrl.set_flux_values([0.0, 0.1, 0.2])
    adapter.ctrl.prepare_run_results()

    # Fresh allocation: a row count but nothing measured yet (all-nan fits).
    summary = _call(adapter, "result.summary", {})["results"]
    assert len(summary) == 1
    entry = summary[0]
    assert entry["name"] == node.name
    assert entry["kind"] == "qubit_freq"
    assert entry["n_flux"] == 3
    assert entry["n_measured"] == 0
    assert entry["fit_summary"]["last_fit_freq"] is None

    # Simulate the worker filling one flux row's fit in place (a consistent
    # mid-run snapshot the main-thread handler reads): n_measured advances.
    result = adapter.ctrl.state.run_results[node.name]
    result.fit_freq[0] = 5000.0
    summary = _call(adapter, "result.summary", {})["results"][0]
    assert summary["n_measured"] == 1
    assert summary["fit_summary"]["last_fit_freq"] == pytest.approx(5000.0)


def test_result_summary_skips_nan_rows():
    # n_measured counts only finite fit rows — an honest "not measured" for nan.
    adapter = _adapter()
    node = adapter.ctrl.add_node_by_type("t1")
    adapter.ctrl.set_flux_values([0.0, 0.1, 0.2, 0.3])
    adapter.ctrl.prepare_run_results()
    result = adapter.ctrl.state.run_results[node.name]
    result.fit_value[1] = 12.0
    result.fit_value[3] = 8.0
    entry = _call(adapter, "result.summary", {})["results"][0]
    assert entry["kind"] == "sweep1d"
    assert entry["n_measured"] == 2
    assert entry["fit_summary"]["last_fit_value"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# resources.versions
# ---------------------------------------------------------------------------


def test_resources_versions_snapshot():
    adapter = _adapter()
    before = _call(adapter, "resources.versions", {})["versions"]
    adapter.ctrl.add_node_by_type("qubit_freq")
    after = _call(adapter, "resources.versions", {})["versions"]
    # adding a node bumps the workflow resource version (optimistic-concurrency)
    assert after != before


def test_versions_snapshot_is_json_friendly():
    adapter = _adapter()
    versions = _call(adapter, "resources.versions", {})["versions"]
    import json

    json.dumps(versions)
    assert all(isinstance(v, int) for v in versions.values())


# ---------------------------------------------------------------------------
# node.cfg knobs are JSON-friendly (no SweepCfg / numpy leaks)
# ---------------------------------------------------------------------------


def test_node_cfg_knobs_are_json_friendly():
    adapter = _adapter()
    for type_name in ("qubit_freq", "lenrabi", "ro_optimize", "t1", "mist"):
        node = adapter.ctrl.add_node_by_type(type_name)
        cfg = _call(adapter, "node.cfg", {"name": node.name})
        import json

        json.dumps(cfg)  # raises if any leaf is not JSON-serialisable
        # sanity: no numpy scalar slipped through
        for v in cfg["knobs"].values():
            assert not isinstance(v, np.generic)
