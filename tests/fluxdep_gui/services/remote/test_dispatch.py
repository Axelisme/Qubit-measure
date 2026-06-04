"""Tests for the fluxdep remote dispatch handlers + method-spec validation.

The remote surface is READ-ONLY: the agent observes a GUI the user drives. So
every handler here is a pure query — there is no load / align / point-pick /
select / fit / export RPC. The tests therefore build the state directly through
the ``Controller`` (the GUI's own command API, exercised by the GUI/controller
tests) and assert that the read handlers report it correctly.

Qt-free: the handlers only touch ``adapter.ctrl`` (a real ``Controller``), so a
tiny stub adapter exercises the whole RPC handler surface without a QApplication
or the socket server.
"""

from __future__ import annotations

import os

import numpy as np
from zcu_tools.fluxdep_gui.controller import Controller
from zcu_tools.fluxdep_gui.services.remote.dispatch import (
    _HANDLERS,
    METHOD_REGISTRY,
)
from zcu_tools.fluxdep_gui.services.remote.method_specs import METHOD_SPECS
from zcu_tools.fluxdep_gui.services.remote.param_spec import validate_params
from zcu_tools.fluxdep_gui.state import FluxDepState, ProjectInfo
from zcu_tools.notebook.persistance import TransitionDict


class _StubAdapter:
    """Minimal stand-in: handlers reach the façade via ``adapter.ctrl`` only."""

    def __init__(self, ctrl: Controller) -> None:
        self.ctrl = ctrl


def _adapter() -> _StubAdapter:
    return _StubAdapter(Controller(FluxDepState()))


def _call(adapter: _StubAdapter, method: str, raw_params: dict) -> dict:
    """Validate params against the spec (as the service does) then dispatch."""
    spec = METHOD_REGISTRY[method]
    params = validate_params(spec.params, raw_params) if spec.params else raw_params
    return dict(spec.handler(adapter, params))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registry coherence
# ---------------------------------------------------------------------------


def test_registry_specs_and_handlers_match():
    assert set(_HANDLERS) == set(METHOD_SPECS)
    assert set(METHOD_REGISTRY) == set(METHOD_SPECS)


def test_registry_is_read_only():
    # Guard against re-introducing a mutating RPC: only these read methods exist.
    assert set(METHOD_SPECS) == {
        "project.info",
        "spectrum.list",
        "selection.pointcloud",
        "fit.result",
        "resources.versions",
        "state.check",
    }


# ---------------------------------------------------------------------------
# project.info
# ---------------------------------------------------------------------------


def test_project_info_reports_state():
    adapter = _adapter()
    adapter.ctrl.setup_project(ProjectInfo(chip_name="Q5_2D", qub_name="Q1"))
    info = _call(adapter, "project.info", {})
    assert info["chip_name"] == "Q5_2D"
    assert info["qub_name"] == "Q1"
    # ProjectInfo derives both paths from chip/qubit in __post_init__
    assert info["result_dir"] == os.path.join("result", "Q5_2D", "Q1")
    assert info["database_path"] == os.path.join("result", "Q5_2D", "Q1")


# ---------------------------------------------------------------------------
# state.check
# ---------------------------------------------------------------------------


def test_state_check_reflects_project_and_spectra(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    adapter = _adapter()
    check = _call(adapter, "state.check", {})
    assert check == {"has_project": False, "spectrum_count": 0, "has_active": False}

    adapter.ctrl.setup_project(ProjectInfo(chip_name="Q5_2D", qub_name="Q1"))
    adapter.ctrl.load_spectrum(filepath, "OneTone")
    check = _call(adapter, "state.check", {})
    assert check["has_project"] is True
    assert check["spectrum_count"] == 1


# ---------------------------------------------------------------------------
# spectrum.list
# ---------------------------------------------------------------------------


def test_spectrum_list_reports_loaded_spectra(spectrum_hdf5):
    filepath, dev_values, freqs_ghz, _ = spectrum_hdf5
    adapter = _adapter()
    name = adapter.ctrl.load_spectrum(filepath, "OneTone")

    listed = _call(adapter, "spectrum.list", {})["spectrums"]
    assert len(listed) == 1
    entry = listed[0]
    assert entry["name"] == name
    assert entry["spec_type"] == "OneTone"
    assert entry["aligned"] is False
    assert entry["points_selected"] is False

    # after alignment + points the flags flip — the agent can watch the user's
    # progress through the pipeline stages
    adapter.ctrl.set_alignment(name, flux_half=0.0, flux_int=5.0)
    adapter.ctrl.set_points(name, np.asarray(dev_values[:3]), np.asarray(freqs_ghz[:3]))
    entry = _call(adapter, "spectrum.list", {})["spectrums"][0]
    assert entry["aligned"] is True
    assert entry["points_selected"] is True


# ---------------------------------------------------------------------------
# selection.pointcloud
# ---------------------------------------------------------------------------


def test_selection_pointcloud_assembles_selected_points(spectrum_hdf5):
    filepath, dev_values, freqs_ghz, _ = spectrum_hdf5
    adapter = _adapter()
    name = adapter.ctrl.load_spectrum(filepath, "OneTone")
    adapter.ctrl.set_alignment(name, flux_half=0.0, flux_int=5.0)
    adapter.ctrl.set_points(name, np.asarray(dev_values[:3]), np.asarray(freqs_ghz[:3]))

    cloud = _call(adapter, "selection.pointcloud", {})
    assert len(cloud["fluxs"]) == 3
    assert len(cloud["freqs"]) == 3
    # freqs come back in GHz (the loader/selection unit), unchanged from the input
    np.testing.assert_allclose(sorted(cloud["freqs"]), sorted(freqs_ghz[:3]))


# ---------------------------------------------------------------------------
# fit.result
# ---------------------------------------------------------------------------


def test_fit_result_reports_inputs_and_result():
    adapter = _adapter()
    adapter.ctrl.set_fit_params(
        "Database/sim/fluxonium.h5",
        (1.0, 12.0),
        (0.5, 3.0),
        (0.1, 2.0),
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    res = _call(adapter, "fit.result", {})
    assert res["has_result"] is False
    assert res["params"] is None
    assert res["database_path"] == "Database/sim/fluxonium.h5"
    assert res["EJb"] == [1.0, 12.0]
    assert res["transitions"] == {"transitions": [[0, 1]]}
    assert res["r_f"] is None
    assert res["sample_f"] is None

    adapter.ctrl.state.set_fit_result((5.0, 1.2, 0.4), best_dist=0.01)
    res = _call(adapter, "fit.result", {})
    assert res["has_result"] is True
    assert res["params"] == {"EJ": 5.0, "EC": 1.2, "EL": 0.4}


# ---------------------------------------------------------------------------
# resources.versions
# ---------------------------------------------------------------------------


def test_resources_versions_snapshot(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    adapter = _adapter()
    before = _call(adapter, "resources.versions", {})["versions"]
    adapter.ctrl.load_spectrum(filepath, "OneTone")
    after = _call(adapter, "resources.versions", {})["versions"]
    # loading bumps at least one resource version (optimistic-concurrency guard)
    assert after != before
