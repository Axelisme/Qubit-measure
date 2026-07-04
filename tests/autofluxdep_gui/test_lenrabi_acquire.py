"""RB-2: lenrabi end-to-end real acquire against the flux-aware MockSoc.

Runs the lenrabi Node's real acquire path (set flux device -> setup_devices ->
ModularProgramV2(Reset, rabi_pulse, Readout).acquire -> fit_rabi) at a few flux
points and asserts a finite, positive pi length comes back -- i.e. the real
acquire produced a fittable Rabi oscillation (not noise / not a constant).

The drive is on resonance: the snapshot's ``qubit_freq`` is the predicted f01 at
each flux (FluxoniumPredictor matching DEFAULT_SIMPARAM), so the Rabi pulse drives
the qubit and the length sweep traces out an oscillation.
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.cfg import SweepValue
from zcu_tools.gui.app.autofluxdep.nodes import lenrabi as lenrabi_mod
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.lenrabi import LenRabiBuilder
from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
from zcu_tools.program.v2 import ModuleCfgFactory, PulseCfg

from ._helpers import (
    ACQUIRE_READOUT,
    connect_mock,
    high_snr_simparams,
    make_acquire_env,
    mock_flux_predictor,
    node_schema,
)

_PARAMS = {
    "qub_ch": 1,
    "qub_nqz": 1,
    "qub_gain": 0.5,
    # start above zero: a const waveform length-sweep needs >= a few FPGA cycles
    "sweep_range": SweepValue(start=0.05, stop=2.0, expts=41),
    "sweep_range_mode": "fixed",
    "drive_gain_mode": "fixed",
    "relax_delay_mode": "fixed",
    "earlystop_snr": None,
    "reps": 1000,
    "rounds": 3,
    "relax_delay": 100.0,
}


def _ml(ctrl):
    ml = ctrl.state.exp_context.ml
    ml.register_waveform(rabi_drive={"style": "const", "length": 1.0})
    return ml


def _drive_pulse() -> PulseCfg:
    pulse = ModuleCfgFactory.from_raw(
        {
            "type": "pulse",
            "ch": 1,
            "nqz": 1,
            "freq": 5135.0,
            "gain": 0.5,
            "waveform": {"style": "const", "length": 1.0},
        }
    )
    assert isinstance(pulse, PulseCfg)
    return pulse


def test_lenrabi_fit_selects_lower_residual(monkeypatch):
    lengths = np.linspace(0.05, 1.0, 11)
    real = np.zeros_like(lengths)

    def fake_fit_rabi(_lengths, _real, /, decay=False, init_phase=None):
        del init_phase
        if decay:
            return 0.4, 0.0, 0.2, 0.0, 1.0, 0.0, real.copy(), None
        return 0.8, 0.0, 0.4, 0.0, 0.5, 0.0, np.ones_like(real), None

    monkeypatch.setattr(lenrabi_mod, "fit_rabi", fake_fit_rabi)

    fit = lenrabi_mod._fit_lenrabi(lengths, real)

    assert fit.pi_length == 0.4
    assert fit.rabi_freq == 1.0


def test_lenrabi_fit_prefers_finite_residual_over_nan_candidate(monkeypatch):
    lengths = np.linspace(0.05, 1.0, 11)
    real = np.zeros_like(lengths)

    def fake_fit_rabi(_lengths, _real, /, decay=False, init_phase=None):
        del init_phase
        if decay:
            return (
                0.4,
                0.0,
                0.2,
                0.0,
                1.0,
                0.0,
                np.full_like(real, np.nan),
                None,
            )
        return 0.8, 0.0, 0.4, 0.0, 0.5, 0.0, real.copy(), None

    monkeypatch.setattr(lenrabi_mod, "fit_rabi", fake_fit_rabi)

    fit = lenrabi_mod._fit_lenrabi(lengths, real)

    assert fit.pi_length == 0.8
    assert fit.rabi_freq == 0.5
    assert np.isfinite(fit.residual)


def test_lenrabi_fit_skips_expected_candidate_runtime_error(monkeypatch):
    lengths = np.linspace(0.05, 1.0, 11)
    real = np.zeros_like(lengths)

    def fake_fit_rabi(_lengths, _real, /, decay=False, init_phase=None):
        del init_phase
        if decay:
            raise RuntimeError("decay candidate did not converge")
        return 0.8, 0.0, 0.4, 0.0, 0.5, 0.0, real.copy(), None

    monkeypatch.setattr(lenrabi_mod, "fit_rabi", fake_fit_rabi)

    fit = lenrabi_mod._fit_lenrabi(lengths, real)

    assert fit.pi_length == 0.8
    assert fit.rabi_freq == 0.5


def test_lenrabi_fit_raises_when_all_candidates_fail(monkeypatch):
    lengths = np.linspace(0.05, 1.0, 11)
    real = np.zeros_like(lengths)

    def fake_fit_rabi(_lengths, _real, /, decay=False, init_phase=None):
        del decay, init_phase
        raise RuntimeError("candidate did not converge")

    monkeypatch.setattr(lenrabi_mod, "fit_rabi", fake_fit_rabi)

    with pytest.raises(RuntimeError, match="decay and non-decay"):
        lenrabi_mod._fit_lenrabi(lengths, real)


def test_lenrabi_fit_propagates_unexpected_exception(monkeypatch):
    lengths = np.linspace(0.05, 1.0, 11)
    real = np.zeros_like(lengths)

    def fake_fit_rabi(_lengths, _real, /, decay=False, init_phase=None):
        del decay, init_phase
        raise TypeError("fit_rabi API changed")

    monkeypatch.setattr(lenrabi_mod, "fit_rabi", fake_fit_rabi)

    with pytest.raises(TypeError, match="API changed"):
        lenrabi_mod._fit_lenrabi(lengths, real)


def test_lenrabi_fit_gate_rejects_boundary_fit_without_feedback_patch():
    lengths = np.linspace(0.05, 1.0, 11)
    real = np.sin(lengths)
    result = Sweep1DResult.allocate(np.array([0.0]), lengths, x_label="pulse length")
    fit = lenrabi_mod._LenRabiFit(
        pi_length=0.95,
        pi2_length=0.2,
        rabi_freq=1.0,
        residual=0.0,
        fit_curve=real.copy(),
    )

    accepted = lenrabi_mod._fill_lenrabi_fit_or_skip(
        result, 0, lengths, real, fit, None
    )
    patch = (
        lenrabi_mod._patch_from_lenrabi_fit(fit, _drive_pulse()) if accepted else None
    )

    assert not accepted
    assert patch is None
    assert np.isnan(result.fit_value[0])


def test_lenrabi_fit_gate_rejects_too_short_pi_length():
    lengths = np.linspace(0.05, 1.0, 11)
    real = np.sin(lengths)
    result = Sweep1DResult.allocate(np.array([0.0]), lengths, x_label="pulse length")
    fit = lenrabi_mod._LenRabiFit(
        pi_length=0.02,
        pi2_length=0.2,
        rabi_freq=1.0,
        residual=0.0,
        fit_curve=real.copy(),
    )

    accepted = lenrabi_mod._fill_lenrabi_fit_or_skip(
        result, 0, lengths, real, fit, None
    )

    assert not accepted
    assert np.isnan(result.fit_value[0])


def test_lenrabi_fit_gate_rejects_large_residual():
    lengths = np.linspace(0.05, 1.0, 11)
    fit_curve = np.sin(lengths)
    real = fit_curve + 0.2
    fit = lenrabi_mod._LenRabiFit(
        pi_length=0.4,
        pi2_length=0.2,
        rabi_freq=1.0,
        residual=0.2,
        fit_curve=fit_curve,
    )

    assert not lenrabi_mod._is_trusted_lenrabi_fit(fit, lengths, real)


def test_lenrabi_patch_emits_coherent_drive_module_pair():
    lengths = np.linspace(0.05, 1.0, 11)
    fit = lenrabi_mod._LenRabiFit(
        pi_length=0.4,
        pi2_length=0.2,
        rabi_freq=1.0,
        residual=0.0,
        fit_curve=np.sin(lengths),
    )

    patch = lenrabi_mod._patch_from_lenrabi_fit(fit, _drive_pulse())

    assert patch.values()["pi_length"] == 0.4
    assert patch.values()["pi2_length"] == 0.2
    assert patch.values()["pi_product"] == pytest.approx(0.2)
    modules = patch.modules()
    assert "pi_pulse" in modules
    assert "pi2_pulse" in modules


@pytest.mark.parametrize("pi2_length", [0.02, float("nan")])
def test_lenrabi_patch_keeps_feedback_but_skips_untrusted_drive_modules(
    pi2_length: float,
):
    lengths = np.linspace(0.05, 1.0, 11)
    fit = lenrabi_mod._LenRabiFit(
        pi_length=0.4,
        pi2_length=pi2_length,
        rabi_freq=1.0,
        residual=0.0,
        fit_curve=np.sin(lengths),
    )

    patch = lenrabi_mod._patch_from_lenrabi_fit(fit, _drive_pulse())

    assert patch.values()["pi_length"] == 0.4
    assert patch.values()["pi_product"] == pytest.approx(0.2)
    assert "rabi_freq" in patch.values()
    assert "pi2_length" not in patch.values()
    assert patch.modules() == {}


def test_lenrabi_acquire_fits_finite_pi_length():
    ctrl = build_core()
    sim_params = high_snr_simparams()
    connect_mock(ctrl, sim_params=sim_params)
    ml = _ml(ctrl)
    predictor = mock_flux_predictor(sim_params)

    builder = LenRabiBuilder()
    schema = node_schema(builder, _PARAMS)
    flux_values = [0.0, 0.0015]
    pis: list[float] = []
    for idx, flux in enumerate(flux_values):
        result = builder.make_init_result(schema, np.asarray(flux_values))
        env = make_acquire_env(
            ctrl, flux=flux, flux_idx=idx, schema=schema, ml=ml, result=result
        )
        f01 = predictor.predict_freq(flux)
        snap = Snapshot(
            {"qubit_freq": float(f01)}, modules={"opt_readout": ACQUIRE_READOUT}
        )
        patch = builder.build_node(env).produce(snap)
        if "pi_length" in patch.values():
            pi_length = float(patch.values()["pi_length"])
            pi2_length = float(patch.values()["pi2_length"])
            modules = patch.modules()
            pi_pulse = ModuleCfgFactory.from_raw(modules["pi_pulse"], ml=ml)
            pi2_pulse = ModuleCfgFactory.from_raw(modules["pi2_pulse"], ml=ml)
            assert isinstance(pi_pulse, PulseCfg)
            assert isinstance(pi2_pulse, PulseCfg)
            assert float(pi_pulse.waveform.length) == pi_length
            assert float(pi2_pulse.waveform.length) == pi2_length
            assert float(pi_pulse.freq) == float(f01)
            assert float(pi2_pulse.freq) == float(f01)
            pis.append(pi_length)

    # at least one flux point produced a finite, positive pi length from the real
    # acquire (a constant / noise signal would fail the fit-quality gate and omit it)
    assert pis, "no flux point fit a pi length from the real acquire"
    assert all(np.isfinite(p) and p > 0.0 for p in pis), (
        f"non-physical pi lengths: {pis}"
    )
