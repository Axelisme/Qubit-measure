"""Tests for dispersive ExportService — write dispersive section, preserve fluxdep_fit."""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.dispersive.services.export import ExportService
from zcu_tools.gui.app.dispersive.state import DispersiveState
from zcu_tools.meta_tool import QubitParams


def test_export_writes_dispersive_and_preserves_fluxdep_fit(params_json):
    path, fit = params_json
    st = DispersiveState()
    st.set_disp_result(g=0.068, bare_rf=5.35, res_dim=4)

    written = ExportService(st).export_params(path)

    assert written == path
    result = QubitParams(path, readonly=True)
    dispersive = result.get_dispersive_fit()
    assert dispersive is not None
    assert dispersive.g == 0.068
    assert dispersive.bare_rf == 5.35
    assert dispersive.timestamp is not None
    # fluxdep_fit section is preserved (not clobbered)
    saved_fit = result.require_fluxdep_fit()
    assert saved_fit.params == (
        fit["params"]["EJ"],
        fit["params"]["EC"],
        fit["params"]["EL"],
    )
    assert saved_fit.flux_period == fit["flux_period"]


def test_export_without_result_fast_fails(params_json):
    path, _fit = params_json
    st = DispersiveState()
    with pytest.raises(RuntimeError, match="no dispersive fit result"):
        ExportService(st).export_params(path)


def test_export_missing_file_fast_fails(tmp_path):
    st = DispersiveState()
    st.set_disp_result(g=0.068, bare_rf=5.35, res_dim=4)
    with pytest.raises(FileNotFoundError, match="must already hold"):
        ExportService(st).export_params(str(tmp_path / "nope.json"))
