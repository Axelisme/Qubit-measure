"""Tests for ConnectionService."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.gui.app.main.services.operation_gate import (
    OperationGate,
)
from zcu_tools.gui.app.main.services.operation_gate import (
    OperationKind as MeasureOpKind,
)
from zcu_tools.gui.app.main.state import ExpContext, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.events import SocChangedPayload
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.ports import OperationConflictError, OperationKind
from zcu_tools.gui.session.services.connection import (
    ConnectionService,
    ConnectMockRequest,
    ConnectRemoteRequest,
    LoadPredictorRequest,
    PredictCurveRequest,
    PredictFreqRequest,
    PredictMatrixCurveRequest,
    PredictorLoadError,
    PredictorNotLoaded,
)
from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM


def _make_svc(gate: OperationGate | None = None) -> ConnectionService:
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    return ConnectionService(
        state, EventBus(), gate or OperationGate(), OperationHandles()
    )


def test_start_connect_mock_emits_finished_and_updates_context(qapp):
    svc = _make_svc()
    loop = QEventLoop()
    svc.connection_finished.connect(loop.quit)
    svc.connection_failed.connect(lambda msg: loop.quit())

    svc.start_connect(ConnectMockRequest())
    loop.exec()

    assert svc.has_soc()
    assert svc._state.exp_context.soccfg is not None


def test_start_connect_mock_soc_carries_default_simparam(qapp):
    """Mock-connect injects DEFAULT_SIMPARAM so the soc yields physical sim data.

    The connection.py mock branch wires DEFAULT_SIMPARAM into make_mock_soc so
    that both "Use MockSoc" and gui_connect_start(kind='mock') return a
    SimEngine-backed soc rather than white noise.  FLUX-AWARE-MOCK copy-on-input:
    the soc keeps an *internal copy* of the injected SimParams (so a later
    set_flux_device never mutates the shared DEFAULT_SIMPARAM singleton), so the
    contract is value-equality, NOT object identity.
    """
    svc = _make_svc()
    loop = QEventLoop()
    svc.connection_finished.connect(loop.quit)
    svc.connection_failed.connect(lambda msg: loop.quit())

    svc.start_connect(ConnectMockRequest())
    loop.exec()

    soc = svc._state.exp_context.soc
    assert soc is not None, "soc must be set after mock connect"
    # Value-equality (not identity): copy-on-input means the soc holds an
    # equivalent copy of DEFAULT_SIMPARAM, not the singleton instance itself.
    # _sim_params is not on SocProtocol (it is MockQickSoc-specific), so we use
    # getattr to satisfy pyright while still asserting the injected params.
    assert hasattr(soc, "_sim_params"), "mock soc must expose _sim_params"
    sim_params = getattr(soc, "_sim_params")
    assert sim_params is not DEFAULT_SIMPARAM  # copy-on-input, not aliased
    assert sim_params == DEFAULT_SIMPARAM


def test_start_connect_mock_sim_params_override_is_honoured(qapp):
    """ConnectMockRequest(sim_params=...) propagates the override into the soc.

    Tests pass a high-snr SimParams to cut wall time (seam documented in
    ConnectMockRequest). This confirms the override path is wired end-to-end:
    the soc must carry the *supplied* params, not DEFAULT_SIMPARAM.
    """
    custom = DEFAULT_SIMPARAM.model_copy(update={"snr": 9999.0})
    svc = _make_svc()
    loop = QEventLoop()
    svc.connection_finished.connect(loop.quit)
    svc.connection_failed.connect(lambda msg: loop.quit())

    svc.start_connect(ConnectMockRequest(sim_params=custom))
    loop.exec()

    soc = svc._state.exp_context.soc
    assert soc is not None, "soc must be set after mock connect"
    sim_params = getattr(soc, "_sim_params", None)
    assert sim_params is not None, "mock soc must expose _sim_params"
    assert sim_params == custom, f"soc carries {sim_params!r}, expected custom override"


def test_connect_bumps_soc_not_context_version(qapp):
    svc = _make_svc()
    ctx_before = svc._state.version.get("context")
    soc_before = svc._state.version.get("soc")
    loop = QEventLoop()
    svc.connection_finished.connect(loop.quit)
    svc.connection_failed.connect(lambda msg: loop.quit())

    svc.start_connect(ConnectMockRequest())
    loop.exec()

    # soc is its own resource; a connect must not spuriously bump context
    # (md/ml content did not change).
    assert svc._state.version.get("soc") == soc_before + 1
    assert svc._state.version.get("context") == ctx_before


def test_predictor_load_clear_does_not_bump_context_version(qapp):
    import dataclasses

    svc = _make_svc()
    ctx_before = svc._state.version.get("context")
    fake = MagicMock()
    svc._state.set_context(dataclasses.replace(svc._state.exp_context, predictor=fake))
    svc.clear_predictor()
    # predictor is not a guarded resource; swapping it must not bump context.
    assert svc._state.version.get("context") == ctx_before


def test_start_connect_rejects_concurrent_calls(qapp):
    gate = OperationGate()
    svc = _make_svc(gate)
    gate.register(1, OperationKind.SOC_CONNECT, owner_id="existing")
    with pytest.raises(OperationConflictError, match="soc_connect is active"):
        svc.start_connect(ConnectMockRequest())


def test_load_predictor_wraps_io_errors(qapp, tmp_path):
    svc = _make_svc()
    with pytest.raises(PredictorLoadError):
        svc.load_predictor(
            LoadPredictorRequest(path=str(tmp_path / "missing.json"), flux_bias=0.0)
        )


def test_predict_freq_without_predictor_raises(qapp):
    svc = _make_svc()
    with pytest.raises(PredictorNotLoaded):
        svc.predict_freq(PredictFreqRequest(value=0.0, transition=(0, 1)))


def test_clear_predictor_resets_state(qapp):
    svc = _make_svc()
    # Inject a fake predictor without going through load_predictor.
    import dataclasses

    fake = MagicMock()
    fake.flux_bias = 0.3
    svc._state.set_context(dataclasses.replace(svc._state.exp_context, predictor=fake))
    svc._predictor_path = "/fake/path.json"

    svc.clear_predictor()
    assert svc.get_predictor() is None
    assert svc.get_predictor_info() is None


def test_start_connect_remote_unsupported_request_raises(qapp):
    svc = _make_svc()

    class Other:
        pass

    with pytest.raises(TypeError, match="Unsupported connect request"):
        svc.start_connect(Other())  # type: ignore[arg-type]


def test_start_connect_remote_failure_emits_failed(qapp, monkeypatch):
    svc = _make_svc()

    # Force make_soc_proxy to raise a connection error inside the worker.
    import zcu_tools.remote as remote

    def fail(ip, port):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(remote, "make_soc_proxy", fail, raising=False)

    loop = QEventLoop()
    errors: list[str] = []
    svc.connection_failed.connect(lambda msg: errors.append(msg) or loop.quit())
    svc.connection_finished.connect(loop.quit)

    svc.start_connect(ConnectRemoteRequest(ip="127.0.0.1", port=7000))
    loop.exec()

    assert errors
    assert "nope" in errors[0]
    assert not svc.is_connect_active()


def test_start_connect_rejected_while_run_active(qapp):
    gate = OperationGate()
    svc = _make_svc(gate)
    gate.register(1, MeasureOpKind.RUN, owner_id="tab")

    with pytest.raises(OperationConflictError, match="run is active"):
        svc.start_connect(ConnectMockRequest())


def test_soc_changed_subscriber_failure_releases_connection_lease(qapp):
    gate = OperationGate()
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    bus = EventBus()
    # A SOC_CHANGED subscriber raising is swallowed + logged by the EventBus; it
    # must NOT propagate out of _finish_success, and the lease is still released
    # (the release is in a finally, independent of subscriber health).
    bus.subscribe(
        SocChangedPayload, MagicMock(side_effect=RuntimeError("render failed"))
    )
    handles = OperationHandles()
    svc = ConnectionService(state, bus, gate, handles)
    # Simulate an in-flight connect: a live handle + registered exclusion.
    token = handles.create()
    gate.register(token, OperationKind.SOC_CONNECT, owner_id="soc")
    svc._active_token = token
    svc._finish_success(MagicMock(), MagicMock())  # no raise — subscriber swallowed
    assert not gate.has_active(OperationKind.SOC_CONNECT)


# ---------------------------------------------------------------------------
# predict_freq_curve tests
# ---------------------------------------------------------------------------


def _inject_fake_predictor(svc: ConnectionService) -> MagicMock:
    """Inject a fake FluxoniumPredictor-like mock into the service state."""

    fake = MagicMock()
    fake.flux_bias = 0.0
    fake.flux_half = 0.5
    fake.flux_period = 1.0
    # params must be a 3-tuple for calculate_energy_vs_flux (not called in these tests)
    fake.params = (1.0, 0.5, 0.5)
    ctx = svc._state.exp_context
    svc._state.set_context(dataclasses.replace(ctx, predictor=fake))
    return fake


def test_predict_freq_curve_no_predictor_raises(qapp):
    svc = _make_svc()
    req = PredictCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((0, 1),),
    )
    with pytest.raises(PredictorNotLoaded):
        svc.predict_freq_curve(req)


def test_predict_freq_curve_empty_transitions_raises(qapp):
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictCurveRequest(values=np.linspace(0.0, 1.0, 10), transitions=())
    with pytest.raises(ValueError, match="transitions must not be empty"):
        svc.predict_freq_curve(req)


def test_predict_freq_curve_negative_level_raises(qapp):
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((-1, 1),),
    )
    with pytest.raises(ValueError, match=">="):
        svc.predict_freq_curve(req)


def test_predict_freq_curve_to_less_than_from_raises(qapp):
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((2, 1),),
    )
    with pytest.raises(ValueError, match="from-level"):
        svc.predict_freq_curve(req)


def test_predict_freq_curve_shape_and_labels(qapp, monkeypatch):
    """predict_freq_curve returns correct shape and labels for multiple transitions.

    We monkeypatch calculate_energy_vs_flux so the test does not need scqubits
    and completes instantly.
    """
    import zcu_tools.simulate.fluxonium.energies as energies_mod

    n_vals = 20
    n_levels = 9
    fake_energies = np.arange(n_vals * n_levels, dtype=np.float64).reshape(
        n_vals, n_levels
    )

    def fake_calc(params, fluxs, cutoff=40, evals_count=20, spectrum_data=None):
        # Return dummy energies matching (n_vals, evals_count).
        e = fake_energies[:, :evals_count]
        return MagicMock(), e

    monkeypatch.setattr(energies_mod, "calculate_energy_vs_flux", fake_calc)

    svc = _make_svc()
    _inject_fake_predictor(svc)

    transitions = ((0, 1), (0, 2), (0, 3), (0, 4))
    values = np.linspace(0.0, 1.0, n_vals)
    req = PredictCurveRequest(values=values, transitions=transitions)
    result = svc.predict_freq_curve(req)

    assert result.freqs_mhz.shape == (len(transitions), n_vals)
    assert result.labels == ("0→1", "0→2", "0→3", "0→4")
    assert len(result.values) == n_vals
    assert len(result.fluxs) == n_vals


def test_predict_freq_curve_matches_single_point(qapp, monkeypatch):
    """Each row of predict_freq_curve must agree with the single-point predict_freq."""
    import zcu_tools.simulate.fluxonium.energies as energies_mod

    n_vals = 5
    n_levels = 9
    # Fixed energies: level i at a given flux = i * (flux_idx + 1)
    fake_base = np.zeros((n_vals, n_levels), dtype=np.float64)
    for v in range(n_vals):
        for lev in range(n_levels):
            fake_base[v, lev] = float(lev * (v + 1))

    def fake_calc(params, fluxs, cutoff=40, evals_count=20, spectrum_data=None):
        return MagicMock(), fake_base[:, :evals_count]

    monkeypatch.setattr(energies_mod, "calculate_energy_vs_flux", fake_calc)

    svc = _make_svc()
    fake = _inject_fake_predictor(svc)

    # Patch predict_freq on the fake predictor to return the same formula.
    def _single_freq(val: float, transition=(0, 1)):
        frm, to = transition
        # Mimic the batch formula: find closest value index.
        values = np.linspace(0.0, 1.0, n_vals)
        idx = int(np.argmin(np.abs(values - val)))
        return float((fake_base[idx, to] - fake_base[idx, frm]) * 1e3)

    fake.predict_freq.side_effect = _single_freq

    transitions = ((0, 1), (0, 2))
    values = np.linspace(0.0, 1.0, n_vals)
    req = PredictCurveRequest(values=values, transitions=transitions)
    result = svc.predict_freq_curve(req)

    # The batch result must match the formula we embedded in fake_base.
    for t_idx, (frm, to) in enumerate(transitions):
        for v_idx in range(n_vals):
            expected = (fake_base[v_idx, to] - fake_base[v_idx, frm]) * 1e3
            assert abs(result.freqs_mhz[t_idx, v_idx] - expected) < 1e-9


def test_get_predictor_info_includes_flux_half_period(qapp):
    """get_predictor_info must now include flux_half and flux_period."""
    svc = _make_svc()
    fake = _inject_fake_predictor(svc)
    fake.flux_half = 0.42
    fake.flux_period = 1.23
    fake.flux_bias = 0.05
    svc._predictor_path = "/fake/path.json"

    info = svc.get_predictor_info()
    assert info is not None
    assert info["flux_half"] == pytest.approx(0.42)
    assert info["flux_period"] == pytest.approx(1.23)
    assert info["flux_bias"] == pytest.approx(0.05)
    assert info["path"] == "/fake/path.json"


# ---------------------------------------------------------------------------
# Real eigensolve alignment: predict_freq_curve vs predictor.predict_freq
# ---------------------------------------------------------------------------


def _make_real_predictor() -> object:
    """Build a real FluxoniumPredictor with physics-plausible Fluxonium parameters.

    Uses moderate EJ/EC/EL values so scqubits converges quickly with cutoff=40.
    flux_half/flux_period/flux_bias chosen so the value→flux affine is non-trivial.
    """
    from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

    params = (4.0, 1.0, 0.5)  # (EJ, EC, EL) in GHz — fast to diagonalise
    return FluxoniumPredictor(params, flux_half=0.3, flux_period=0.8, flux_bias=0.0)


def _inject_real_predictor(svc: ConnectionService) -> object:
    """Replace the fake predictor in svc state with a real FluxoniumPredictor."""
    import dataclasses

    predictor = _make_real_predictor()
    ctx = svc._state.exp_context
    svc._state.set_context(dataclasses.replace(ctx, predictor=predictor))
    return predictor


@pytest.mark.slow
def test_predict_freq_curve_aligns_with_single_point_real_eigensolve(qapp):
    """predict_freq_curve values must match predictor.predict_freq at sampled points.

    This test exercises the real scqubits diagonalisation with cutoff=40 to pin
    the cutoff alignment introduced in connection.py (item A of the code review).
    Uses only 5 grid points to keep runtime short while still catching drift.
    """
    svc = _make_svc()
    predictor = _inject_real_predictor(svc)

    # 5-point grid across a sub-Φ₀ window (avoids near-degenerate half-flux point).
    values = np.linspace(0.05, 0.35, 5)
    transition = (0, 1)
    req = PredictCurveRequest(values=values, transitions=(transition,))
    result = svc.predict_freq_curve(req)

    curve_row = result.freqs_mhz[0]  # shape (5,)

    # Compare each grid point against the single-point path.
    for i, val in enumerate(values):
        expected = predictor.predict_freq(float(val), transition)  # type: ignore[attr-defined]
        assert np.isclose(curve_row[i], expected, rtol=1e-4), (
            f"value={val:.4f}: curve={curve_row[i]:.4f} MHz, single={expected:.4f} MHz"
        )


# ---------------------------------------------------------------------------
# predict_matrix_element_curve tests
# ---------------------------------------------------------------------------


def test_predict_matrix_element_curve_no_predictor_raises(qapp):
    svc = _make_svc()
    req = PredictMatrixCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((0, 1),),
        operator="n",
    )
    with pytest.raises(PredictorNotLoaded):
        svc.predict_matrix_element_curve(req)


def test_predict_matrix_element_curve_empty_transitions_raises(qapp):
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictMatrixCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=(),
        operator="n",
    )
    with pytest.raises(ValueError, match="transitions must not be empty"):
        svc.predict_matrix_element_curve(req)


def test_predict_matrix_element_curve_negative_level_raises(qapp):
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictMatrixCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((-1, 1),),
        operator="n",
    )
    with pytest.raises(ValueError, match=">="):
        svc.predict_matrix_element_curve(req)


def test_predict_matrix_element_curve_to_less_than_from_raises(qapp):
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictMatrixCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((2, 1),),
        operator="n",
    )
    with pytest.raises(ValueError, match="from-level"):
        svc.predict_matrix_element_curve(req)


def test_predict_matrix_element_curve_shape_and_labels_n_oper(qapp, monkeypatch):
    """predict_matrix_element_curve returns correct shape and labels (n operator)."""
    import zcu_tools.simulate.fluxonium.matrix_element as mat_mod

    n_vals = 15

    def fake_n_oper_vs_flux(params, fluxs, return_dim=4, spectrum_data=None):
        # Return complex opers of shape (n_vals, return_dim, return_dim)
        opers = np.ones((n_vals, return_dim, return_dim), dtype=np.complex128) * (
            0.3 + 0.1j
        )
        return MagicMock(), opers

    monkeypatch.setattr(mat_mod, "calculate_n_oper_vs_flux", fake_n_oper_vs_flux)

    svc = _make_svc()
    _inject_fake_predictor(svc)

    transitions = ((0, 1), (0, 2), (0, 3), (1, 4))
    values = np.linspace(0.0, 1.0, n_vals)
    req = PredictMatrixCurveRequest(
        values=values, transitions=transitions, operator="n"
    )
    result = svc.predict_matrix_element_curve(req)

    assert result.mags.shape == (len(transitions), n_vals)
    assert result.labels == ("0→1", "0→2", "0→3", "1→4")
    assert len(result.values) == n_vals
    assert len(result.fluxs) == n_vals
    # All magnitudes are abs(0.3+0.1j) ≈ 0.3162
    assert np.allclose(result.mags, abs(0.3 + 0.1j))


def test_predict_matrix_element_curve_phi_operator(qapp, monkeypatch):
    """predict_matrix_element_curve dispatches to phi operator when requested."""
    import zcu_tools.simulate.fluxonium.matrix_element as mat_mod

    n_vals = 10
    call_log: list[str] = []

    def fake_n(params, fluxs, return_dim=4, spectrum_data=None):
        call_log.append("n")
        return MagicMock(), np.zeros(
            (n_vals, return_dim, return_dim), dtype=np.complex128
        )

    def fake_phi(params, fluxs, return_dim=4, spectrum_data=None):
        call_log.append("phi")
        return MagicMock(), np.ones(
            (n_vals, return_dim, return_dim), dtype=np.complex128
        ) * 0.5

    monkeypatch.setattr(mat_mod, "calculate_n_oper_vs_flux", fake_n)
    monkeypatch.setattr(mat_mod, "calculate_phi_oper_vs_flux", fake_phi)

    svc = _make_svc()
    _inject_fake_predictor(svc)

    req = PredictMatrixCurveRequest(
        values=np.linspace(0.0, 1.0, n_vals),
        transitions=((0, 1),),
        operator="phi",
    )
    result = svc.predict_matrix_element_curve(req)

    assert "phi" in call_log
    assert "n" not in call_log
    assert np.allclose(result.mags, 0.5)


def test_predict_matrix_element_curve_high_level_transition_does_not_raise(
    qapp, monkeypatch
):
    """Transitions with levels > 1 (e.g. 0→3) must not raise.

    This verifies the key goal: we bypass FluxoniumPredictor.predict_matrix_element's
    hard level<=1 cap by calling calculate_*_oper_vs_flux directly with needed_dim.
    """
    import zcu_tools.simulate.fluxonium.matrix_element as mat_mod

    n_vals = 8
    captured_dim: list[int] = []

    def fake_n(params, fluxs, return_dim=4, spectrum_data=None):
        captured_dim.append(return_dim)
        # Large enough matrix for the test transitions.
        opers = np.zeros((n_vals, return_dim, return_dim), dtype=np.complex128)
        # Set distinct non-zero values on each requested off-diagonal.
        for i in range(return_dim):
            for j in range(return_dim):
                if i != j:
                    opers[:, i, j] = 0.1 * (i + 1) + 0.01j * (j + 1)
        return MagicMock(), opers

    monkeypatch.setattr(mat_mod, "calculate_n_oper_vs_flux", fake_n)

    svc = _make_svc()
    _inject_fake_predictor(svc)

    transitions = (
        (0, 1),
        (0, 3),
    )  # level 3 would be rejected by predict_matrix_element
    req = PredictMatrixCurveRequest(
        values=np.linspace(0.0, 1.0, n_vals),
        transitions=transitions,
        operator="n",
    )
    result = svc.predict_matrix_element_curve(req)

    # Confirm needed_dim was passed as 4 (max level 3 + 1).
    assert captured_dim[-1] == 4
    assert result.mags.shape == (2, n_vals)
    # (0,3) magnitude must be non-zero (would be zero if level cap was enforced).
    assert np.all(result.mags[1] > 0)
