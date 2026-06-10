"""Integration test: the full dispersive pipeline through the Controller.

Exercises load_fit_inputs → load_onetone → preprocess → predict → autofit → export,
with scqubits-backed calls monkeypatched to cheap stubs, asserting the EventBus
fires and the predictor cache is bound to the right (params, flux-axis).
"""

from __future__ import annotations

import json

import numpy as np
import zcu_tools.gui.app.dispersive.services.predict as predict_mod
from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.event_bus import (
    DispFitChangedPayload,
    EventBus,
    FitInputsLoadedPayload,
    OnetoneLoadedPayload,
    PreprocessChangedPayload,
)
from zcu_tools.gui.app.dispersive.state import DispersiveState
from zcu_tools.gui.project import ProjectInfo


def _stub(params, fluxs, bare_rf, g, *, progress=False, res_dim=4, **kw):
    rf0 = np.full(len(fluxs), 5.5 + 2.0 * (g - 0.06))
    rf1 = np.full(len(fluxs), 5.9)
    return rf0, rf1


def _params_json(tmp_path):
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
            },
            f,
        )
    return path


def test_full_pipeline(monkeypatch, tmp_path, onetone_hdf5):
    monkeypatch.setattr(predict_mod, "calculate_dispersive_vs_flux_fast", _stub)
    onetone_path, *_ = onetone_hdf5
    params_path = _params_json(tmp_path)

    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
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
    ctrl.load_fit_inputs(params_path)
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

    # 4. predict (cached, stubbed) — the tune worker's compute
    rf = ctrl.predict_dispersive(0.06, 5.3, return_dim=2)
    assert len(rf) == 2

    # 5. accept the tuning — the manual g/bare_rf IS the final fit (no auto-fit)
    ctrl.set_manual_fit(0.06, 5.3, res_dim=4)
    assert state.disp_fit.has_result
    assert state.disp_fit.g == 0.06 and state.disp_fit.bare_rf == 5.3
    assert "DispFitChangedPayload" in events

    # 6. export preserves fluxdep_fit
    ctrl.export_params(params_path)
    from zcu_tools.notebook.persistance import load_result

    saved = load_result(params_path)
    assert saved.get("dispersive") is not None
    assert saved.get("fluxdep_fit") is not None


def test_predictor_rebuilt_when_inputs_change(monkeypatch, tmp_path, onetone_hdf5):
    monkeypatch.setattr(predict_mod, "calculate_dispersive_vs_flux_fast", _stub)
    onetone_path, *_ = onetone_hdf5
    params_path = _params_json(tmp_path)
    state = DispersiveState(ProjectInfo(chip_name="C", qub_name="Q1"))
    ctrl = Controller(state)

    ctrl.load_fit_inputs(params_path)
    ctrl.load_onetone(onetone_path)
    ctrl.record_preprocess(ctrl.compute_preprocess())
    p1 = ctrl._predictor()
    # re-recording preprocess (new object) rebuilds the predictor
    ctrl.record_preprocess(ctrl.compute_preprocess())
    p2 = ctrl._predictor()
    assert p1 is not p2
