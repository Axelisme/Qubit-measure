"""Tests for the dispersive remote dispatch handlers + method-spec validation.

The remote surface is READ-ONLY: the agent observes a GUI the user drives. Every
handler is a pure query — no load / preprocess / tune / fit / export RPC. The tests
build state directly through the ``Controller`` and assert the read handlers report
it correctly. Qt-free: a tiny stub adapter exercises the handlers without a
QApplication or the socket server.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.services.remote.dispatch import (
    _HANDLERS,
    METHOD_REGISTRY,
)
from zcu_tools.gui.app.dispersive.services.remote.method_specs import METHOD_SPECS
from zcu_tools.gui.app.dispersive.state import (
    DispersiveState,
    FluxoniumInputs,
    OnetoneEntry,
    PreprocessResult,
    ProjectInfo,
)
from zcu_tools.notebook.persistance import SpectrumData


class _StubAdapter:
    """Minimal stand-in: handlers reach the façade via ``adapter.ctrl`` only."""

    def __init__(self, ctrl: Controller) -> None:
        self.ctrl = ctrl


def _adapter(state: DispersiveState | None = None) -> _StubAdapter:
    return _StubAdapter(Controller(state if state is not None else DispersiveState()))


def _call(adapter: _StubAdapter, method: str) -> dict:
    """Run a read handler through the registry (stub adapter — no Qt/socket)."""
    return dict(METHOD_REGISTRY[method].handler(adapter, {}))  # type: ignore[arg-type]


def _inputs() -> FluxoniumInputs:
    return FluxoniumInputs(
        params=(4.0, 1.0, 0.5),
        flux_half=0.5,
        flux_int=1.0,
        flux_period=2.0,
        bare_rf_seed=5.3,
    )


# --- registry integrity ------------------------------------------------


def test_every_spec_has_a_handler_and_vice_versa():
    assert set(_HANDLERS) == set(METHOD_SPECS)
    assert set(METHOD_REGISTRY) == set(METHOD_SPECS)


def test_method_set_is_read_only():
    # No mutating verbs in the method names — pure queries only.
    for method in METHOD_SPECS:
        assert any(
            method.endswith(suffix)
            for suffix in (".info", ".status", ".result", ".check", ".versions")
        ), method


# --- handler behaviour -------------------------------------------------


def test_state_check_reports_pipeline_progress():
    state = DispersiveState(ProjectInfo(chip_name="ChipA", qub_name="Q1"))
    adapter = _adapter(state)
    out = _call(adapter, "state.check")
    assert out == {
        "has_project": True,
        "has_fit_inputs": False,
        "has_onetone": False,
        "has_preprocess": False,
        "has_result": False,
    }

    state.set_fit_inputs(_inputs())
    out = _call(adapter, "state.check")
    assert out["has_fit_inputs"] is True


def test_state_check_unknown_project_is_false():
    out = _call(_adapter(), "state.check")
    assert out["has_project"] is False  # default unknown_* placeholders


def test_project_info_handler():
    state = DispersiveState(ProjectInfo(chip_name="ChipA", qub_name="Q1"))
    out = _call(_adapter(state), "project.info")
    assert out["chip_name"] == "ChipA"
    assert out["qub_name"] == "Q1"


def test_fit_inputs_info_none_then_loaded():
    state = DispersiveState()
    adapter = _adapter(state)
    out = _call(adapter, "fit_inputs.info")
    assert out["has_inputs"] is False and out["params"] is None

    state.set_fit_inputs(_inputs())
    out = _call(adapter, "fit_inputs.info")
    assert out["has_inputs"] is True
    assert out["params"] == {"EJ": 4.0, "EC": 1.0, "EL": 0.5}
    assert out["bare_rf_seed"] == 5.3


def test_preprocess_status_handler():
    state = DispersiveState()
    adapter = _adapter(state)
    out = _call(adapter, "preprocess.status")
    assert out["has_preprocess"] is False

    state.set_preprocess(
        PreprocessResult(
            sp_fluxs=np.zeros(4),
            sp_freqs=np.zeros(6),
            norm_phases=np.zeros((4, 6)),
            edelays=np.zeros(4),
            edelay=1.5,
            signature=("x",),
        )
    )
    out = _call(adapter, "preprocess.status")
    assert out == {"has_preprocess": True, "n_flux": 4, "n_freq": 6, "edelay": 1.5}


def test_fit_result_handler():
    state = DispersiveState()
    adapter = _adapter(state)
    out = _call(adapter, "fit.result")
    assert out["has_result"] is False and out["g"] is None

    state.set_disp_result(g=0.068, bare_rf=5.35, res_dim=5, step=2)
    out = _call(adapter, "fit.result")
    assert out["has_result"] is True
    assert out["g"] == 0.068 and out["bare_rf"] == 5.35
    assert out["res_dim"] == 5 and out["step"] == 2


def test_resources_versions_handler():
    out = _call(_adapter(), "resources.versions")
    assert "versions" in out and isinstance(out["versions"], dict)


# --- onetone entry presence (used by state.check) ----------------------


def test_state_check_has_onetone():
    state = DispersiveState()
    state.set_fit_inputs(_inputs())
    e = np.linspace(0.0, 1.0, 3).astype(np.float64)
    state.set_onetone(
        OnetoneEntry(
            name="r1",
            raw=SpectrumData(
                dev_values=e.copy(),
                fluxs=e.copy(),
                freqs=e.copy(),
                signals=np.zeros((3, 3), dtype=np.complex128),
            ),
        )
    )
    out = _call(_adapter(state), "state.check")
    assert out["has_onetone"] is True
