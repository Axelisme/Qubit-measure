"""Tests for dispersive ProjectService — read fluxdep_fit from params.json."""

from __future__ import annotations

import json

import pytest
from zcu_tools.gui.app.dispersive.services.project import ProjectService
from zcu_tools.gui.app.dispersive.state import DEFAULT_BARE_RF, DispersiveState


def test_load_fit_inputs_reads_params_and_seeds_bare_rf(params_json):
    path, fit = params_json
    st = DispersiveState()
    inputs = ProjectService(st).load_fit_inputs(path)

    assert inputs.params == (4.0, 1.0, 0.5)
    assert inputs.flux_half == 0.5
    assert inputs.flux_period == 2.0
    assert inputs.bare_rf_seed == 5.3  # from plot_transitions.r_f
    assert st.fit_inputs is inputs
    assert st.disp_fit.bare_rf == 5.3  # seeded into tuning state


def test_bare_rf_priority_prefers_dispersive_section(tmp_path):
    path = str(tmp_path / "params.json")
    with open(path, "w", encoding="utf8") as f:
        json.dump(
            {
                "name": "Q1",
                "fluxdep_fit": {
                    "params": {"EJ": 4.0, "EC": 1.0, "EL": 0.5},
                    "flux_half": 0.5,
                    "flux_int": 1.0,
                    "flux_period": 2.0,
                    "plot_transitions": {"r_f": 5.3},
                },
                "dispersive": {"g": 0.06, "bare_rf": 5.9},
            },
            f,
        )
    st = DispersiveState()
    inputs = ProjectService(st).load_fit_inputs(path)
    assert inputs.bare_rf_seed == 5.9  # dispersive section wins over r_f


def test_bare_rf_falls_back_to_default(tmp_path):
    path = str(tmp_path / "params.json")
    with open(path, "w", encoding="utf8") as f:
        json.dump(
            {
                "name": "Q1",
                "fluxdep_fit": {
                    "params": {"EJ": 4.0, "EC": 1.0, "EL": 0.5},
                    "flux_half": 0.5,
                    "flux_int": 1.0,
                    "flux_period": 2.0,
                    "plot_transitions": {},  # no r_f
                },
            },
            f,
        )
    st = DispersiveState()
    inputs = ProjectService(st).load_fit_inputs(path)
    assert inputs.bare_rf_seed == DEFAULT_BARE_RF


def test_missing_file_fast_fails(tmp_path):
    st = DispersiveState()
    with pytest.raises(FileNotFoundError, match="run fluxdep-gui first"):
        ProjectService(st).load_fit_inputs(str(tmp_path / "nope.json"))


def test_missing_fluxdep_fit_fast_fails(tmp_path):
    path = str(tmp_path / "params.json")
    with open(path, "w", encoding="utf8") as f:
        json.dump({"name": "Q1"}, f)
    st = DispersiveState()
    with pytest.raises(ValueError, match="no 'fluxdep_fit'"):
        ProjectService(st).load_fit_inputs(path)


def test_default_params_path_uses_result_dir():
    from zcu_tools.gui.app.dispersive.services.project import default_params_path

    assert default_params_path("result/ChipA/Q1").endswith("params.json")
