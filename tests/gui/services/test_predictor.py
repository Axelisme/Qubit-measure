"""Tests for PredictorService."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.gui.app.main.state import ExpContext, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.services.predictor import (
    PredictCurveRequest,
    PredictFreqRequest,
    PredictMatrixCurveRequest,
    PredictorLoadError,
    PredictorNotLoaded,
    PredictorService,
    SetModelParamsRequest,
    read_fluxdep_fit_params,
)


def _make_svc() -> PredictorService:
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    return PredictorService(state, EventBus())


def _inject_fake_predictor(svc: PredictorService) -> MagicMock:
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


# ---------------------------------------------------------------------------
# Predictor load / clear / version invariant
# ---------------------------------------------------------------------------


def test_predictor_load_clear_does_not_bump_context_version():
    svc = _make_svc()
    ctx_before = svc._state.version.get("context")
    fake = MagicMock()
    svc._state.set_context(dataclasses.replace(svc._state.exp_context, predictor=fake))
    svc.clear_predictor()
    # predictor is not a guarded resource; swapping it must not bump context.
    assert svc._state.version.get("context") == ctx_before


def test_load_predictor_wraps_io_errors(tmp_path):
    svc = _make_svc()
    with pytest.raises(PredictorLoadError):
        from zcu_tools.gui.session.services.predictor import LoadPredictorRequest

        svc.load_predictor(
            LoadPredictorRequest(path=str(tmp_path / "missing.json"), flux_bias=0.0)
        )


def test_predict_freq_without_predictor_raises():
    svc = _make_svc()
    with pytest.raises(PredictorNotLoaded):
        svc.predict_freq(PredictFreqRequest(value=0.0, transition=(0, 1)))


def test_clear_predictor_resets_state():
    svc = _make_svc()
    # Inject a fake predictor without going through load_predictor.
    fake = MagicMock()
    fake.flux_bias = 0.3
    svc._state.set_context(dataclasses.replace(svc._state.exp_context, predictor=fake))
    svc._predictor_path = "/fake/path.json"

    svc.clear_predictor()
    assert svc.get_predictor() is None
    assert svc.get_predictor_info() is None


# ---------------------------------------------------------------------------
# predict_freq_curve tests
# ---------------------------------------------------------------------------


def test_predict_freq_curve_no_predictor_raises():
    svc = _make_svc()
    req = PredictCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((0, 1),),
    )
    with pytest.raises(PredictorNotLoaded):
        svc.predict_freq_curve(req)


def test_predict_freq_curve_empty_transitions_raises():
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictCurveRequest(values=np.linspace(0.0, 1.0, 10), transitions=())
    with pytest.raises(ValueError, match="transitions must not be empty"):
        svc.predict_freq_curve(req)


def test_predict_freq_curve_negative_level_raises():
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((-1, 1),),
    )
    with pytest.raises(ValueError, match=">="):
        svc.predict_freq_curve(req)


def test_predict_freq_curve_to_less_than_from_raises():
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((2, 1),),
    )
    with pytest.raises(ValueError, match="from-level"):
        svc.predict_freq_curve(req)


def test_predict_freq_curve_shape_and_labels(monkeypatch):
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


def test_predict_freq_curve_matches_single_point(monkeypatch):
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


def test_get_predictor_info_includes_flux_half_period():
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


def test_get_predictor_info_includes_energies():
    """get_predictor_info exposes EJ/EC/EL from predictor.params for read-back."""
    svc = _make_svc()
    fake = _inject_fake_predictor(svc)
    fake.params = (4.2, 1.1, 0.7)

    info = svc.get_predictor_info()
    assert info is not None
    assert info["EJ"] == pytest.approx(4.2)
    assert info["EC"] == pytest.approx(1.1)
    assert info["EL"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# set_model_params: in-memory build+install (no file)
# ---------------------------------------------------------------------------


def test_set_model_params_builds_and_installs_predictor():
    """set_model_params installs a real FluxoniumPredictor; info reads it back.

    Exercises the in-memory install seam (no params.json): the predictor object
    is constructed straight from typed energies, so path stays None.
    """
    svc = _make_svc()
    req = SetModelParamsRequest(
        EJ=4.0,
        EC=1.0,
        EL=1.0,
        flux_half=0.3,
        flux_period=0.8,
        flux_bias=0.1,
    )
    svc.set_model_params(req)

    predictor = svc.get_predictor()
    assert predictor is not None
    assert predictor.params == (4.0, 1.0, 1.0)

    info = svc.get_predictor_info()
    assert info is not None
    assert info["EJ"] == pytest.approx(4.0)
    assert info["EC"] == pytest.approx(1.0)
    assert info["EL"] == pytest.approx(1.0)
    assert info["flux_half"] == pytest.approx(0.3)
    assert info["flux_period"] == pytest.approx(0.8)
    assert info["flux_bias"] == pytest.approx(0.1)
    # In-memory install has no backing file.
    assert info["path"] is None


def test_set_model_params_emits_predictor_changed():
    """Installing via set_model_params must fan a PredictorChangedPayload."""
    from zcu_tools.gui.session.events import PredictorChangedPayload

    svc = _make_svc()
    seen: list[object] = []
    svc._bus.subscribe(PredictorChangedPayload, lambda p: seen.append(p))

    svc.set_model_params(
        SetModelParamsRequest(EJ=4.0, EC=1.0, EL=1.0, flux_half=0.3, flux_period=0.8)
    )
    assert len(seen) == 1


def test_set_model_params_zero_period_raises():
    """flux_period == 0 makes the value<->flux affine singular — fast-fail."""
    svc = _make_svc()
    with pytest.raises(PredictorLoadError, match="flux_period"):
        svc.set_model_params(
            SetModelParamsRequest(
                EJ=4.0, EC=1.0, EL=1.0, flux_half=0.3, flux_period=0.0
            )
        )
    # Nothing installed on the failure path.
    assert svc.get_predictor() is None


# ---------------------------------------------------------------------------
# read_fluxdep_fit_params: pure params.json -> typed request query
# ---------------------------------------------------------------------------


def _write_params_json(path, *, with_fluxdep: bool = True) -> None:
    import json

    payload: dict = {"name": "fake_qubit"}
    if with_fluxdep:
        payload["fluxdep_fit"] = {
            "params": {"EJ": 4.5, "EC": 1.2, "EL": 0.9},
            "flux_half": 0.31,
            "flux_int": 0.71,
            "flux_period": 0.8,
            "plot_transitions": {},
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_fluxdep_fit_params_returns_request(tmp_path):
    """read_fluxdep_fit_params parses fluxdep_fit into a SetModelParamsRequest."""
    p = tmp_path / "params.json"
    _write_params_json(p)

    req = read_fluxdep_fit_params(str(p))
    assert isinstance(req, SetModelParamsRequest)
    assert req.EJ == pytest.approx(4.5)
    assert req.EC == pytest.approx(1.2)
    assert req.EL == pytest.approx(0.9)
    assert req.flux_half == pytest.approx(0.31)
    assert req.flux_period == pytest.approx(0.8)
    # flux_int is alignment-only and must NOT leak into the model request;
    # flux_bias defaults to 0.0 (the file carries no per-measurement bias).
    assert req.flux_bias == 0.0


def test_read_fluxdep_fit_params_missing_file_raises(tmp_path):
    with pytest.raises(PredictorLoadError, match="Failed to read params file"):
        read_fluxdep_fit_params(str(tmp_path / "missing.json"))


def test_read_fluxdep_fit_params_no_fluxdep_section_raises(tmp_path):
    p = tmp_path / "params.json"
    _write_params_json(p, with_fluxdep=False)
    with pytest.raises(PredictorLoadError, match="fluxdep_fit"):
        read_fluxdep_fit_params(str(p))


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


def _inject_real_predictor(svc: PredictorService) -> object:
    """Replace the fake predictor in svc state with a real FluxoniumPredictor."""
    predictor = _make_real_predictor()
    ctx = svc._state.exp_context
    svc._state.set_context(dataclasses.replace(ctx, predictor=predictor))
    return predictor


@pytest.mark.slow
def test_predict_freq_curve_aligns_with_single_point_real_eigensolve():
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


def test_predict_matrix_element_curve_no_predictor_raises():
    svc = _make_svc()
    req = PredictMatrixCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((0, 1),),
        operator="n",
    )
    with pytest.raises(PredictorNotLoaded):
        svc.predict_matrix_element_curve(req)


def test_predict_matrix_element_curve_empty_transitions_raises():
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictMatrixCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=(),
        operator="n",
    )
    with pytest.raises(ValueError, match="transitions must not be empty"):
        svc.predict_matrix_element_curve(req)


def test_predict_matrix_element_curve_negative_level_raises():
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictMatrixCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((-1, 1),),
        operator="n",
    )
    with pytest.raises(ValueError, match=">="):
        svc.predict_matrix_element_curve(req)


def test_predict_matrix_element_curve_to_less_than_from_raises():
    svc = _make_svc()
    _inject_fake_predictor(svc)
    req = PredictMatrixCurveRequest(
        values=np.linspace(0.0, 1.0, 10),
        transitions=((2, 1),),
        operator="n",
    )
    with pytest.raises(ValueError, match="from-level"):
        svc.predict_matrix_element_curve(req)


def test_predict_matrix_element_curve_shape_and_labels_n_oper(monkeypatch):
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


def test_predict_matrix_element_curve_phi_operator(monkeypatch):
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
    monkeypatch,
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
