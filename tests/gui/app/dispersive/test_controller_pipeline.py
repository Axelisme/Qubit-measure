"""Integration tests for the public dispersive Controller pipeline."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.event_bus import (
    DispFitChangedPayload,
    EventBus,
    FitInputsLoadedPayload,
    OnetoneLoadedPayload,
    PreprocessChangedPayload,
)
from zcu_tools.gui.app.dispersive.state import DispersiveState
from zcu_tools.resources.dispersive_file import load_dispersive_file


def test_full_pipeline(analysis_target, onetone_hdf5):
    onetone_path, *_ = onetone_hdf5
    state = DispersiveState(analysis_target)
    bus = EventBus()
    events = []
    for ptype in (
        FitInputsLoadedPayload,
        OnetoneLoadedPayload,
        PreprocessChangedPayload,
        DispFitChangedPayload,
    ):
        bus.subscribe(ptype, lambda p: events.append(type(p).__name__))
    ctrl = Controller(state, bus)

    # 1. load fit inputs
    ctrl.load_fit_inputs("fit.json")
    assert state.fit_inputs is not None
    assert "FitInputsLoadedPayload" in events

    # 2. load onetone
    ctrl.load_onetone(onetone_path)
    assert state.onetone is not None
    assert "OnetoneLoadedPayload" in events

    # 3. preprocess (real pipeline)
    result = ctrl.compute_preprocess()
    ctrl.record_preprocess(result)
    assert state.preprocess is not None
    assert "PreprocessChangedPayload" in events

    # 4. predict over the full preprocessed axis
    rf = ctrl.predict_dispersive(0.06, 5.3, return_dim=2)
    assert len(rf) == 2
    assert all(line.shape == result.sp_fluxs.shape for line in rf)

    # 5. accept the tuning — the manual g/bare_rf IS the final fit (no auto-fit)
    ctrl.set_manual_fit(0.06, 5.3, res_dim=4)
    assert state.disp_fit.has_result
    assert state.disp_fit.g == 0.06 and state.disp_fit.bare_rf == 5.3
    assert "DispFitChangedPayload" in events

    # 6. export publishes an entry-scoped dispersive artifact.
    ctrl.export_params()
    saved = load_dispersive_file(analysis_target.resource.dispersive_path / "q1.json")
    assert saved.params.g == 0.06
    assert saved.params.bare_rf == 5.3


def test_predict_flux_axis_updates_when_preprocess_changes(
    analysis_target, onetone_hdf5
):
    onetone_path, *_ = onetone_hdf5
    state = DispersiveState(analysis_target)
    ctrl = Controller(state)
    ctrl.load_fit_inputs("fit.json")
    ctrl.load_onetone(onetone_path)
    result = ctrl.preprocess()
    np.testing.assert_array_equal(ctrl.predict_flux_axis(), result.sp_fluxs)

    updated = replace(result, sp_fluxs=result.sp_fluxs + 0.25)
    ctrl.record_preprocess(updated)
    np.testing.assert_array_equal(ctrl.predict_flux_axis(), updated.sp_fluxs)
