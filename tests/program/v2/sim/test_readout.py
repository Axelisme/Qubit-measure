"""Tests for sim/readout.py — the dispersive readout (physics -> IQ) model.

Covers:
- resonator_freqs: dressed rf_g/rf_e near bare_rf with a non-zero dispersive shift.
- Q3 fallback: DressedLabelingError degrades to (bare_rf, bare_rf) + warning.
- mixed_signal: takes the (rf_g, rf_e) dressed frequencies directly (the engine
  resolves them once); p_e endpoints match S21(rf_g)/S21(rf_e), midpoint is the
  mean, and an onetone sweep shows a resonance dip near rf_g.
- Fast-fail: p_e outside [0, 1] raises.
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.program.v2.sim import readout
from zcu_tools.program.v2.sim.readout import (
    mixed_signal,
    resonator_freqs,
    s21,
)
from zcu_tools.simulate.fluxonium.dispersive import DressedLabelingError

from zcu_tools.program.v2.sim import SimParams  # isort: skip

# Physically reasonable fluxonium + hanger parameters reused across tests.
# Qi=50000 >> Ql=5000 → deep dip (dip depth = 1 - Ql/Qi = 0.9).
_SIM = SimParams(
    EJ=8.5,
    EC=1.0,
    EL=0.5,
    flux_period=0.002,
    flux_half=0.001,
    T1=50.0,
    T2=30.0,
    T2_star=30.0,  # T2_star == T2 => gamma=0 (pure homogeneous; preserves existing physics)
    bare_rf=7.2,
    g=0.08,
    Ql=5000.0,
    Qi=50000.0,
    snr=10.0,
    pi_gain_len=0.4,
)


class TestResonatorFreqs:
    def test_dispersive_shift_present(self) -> None:
        rf_g, rf_e = resonator_freqs(_SIM, flux=0.3)
        # Both dressed frequencies sit near the bare resonator (dispersive regime).
        assert abs(rf_g - _SIM.bare_rf) < 0.1
        assert abs(rf_e - _SIM.bare_rf) < 0.1
        # A real dispersive shift means the two states differ.
        assert rf_g != rf_e
        assert abs(rf_e - rf_g) > 1e-4  # at least ~0.1 MHz

    def test_returns_python_floats(self) -> None:
        rf_g, rf_e = resonator_freqs(_SIM, flux=0.25)
        assert isinstance(rf_g, float)
        assert isinstance(rf_e, float)


class TestDressedLabelingFallback:
    def test_fallback_to_bare_rf_with_warning(self, monkeypatch) -> None:
        # Force the dispersive computation to raise so the deterministic
        # degradation path is exercised regardless of parameter regime.
        def _raise(*args, **kwargs):
            raise DressedLabelingError("forced ambiguity for test")

        monkeypatch.setattr(readout, "calculate_dispersive_vs_flux_fast", _raise)

        with pytest.warns(UserWarning, match="dispersive labeling ambiguous"):
            rf_g, rf_e = resonator_freqs(_SIM, flux=0.42)

        assert rf_g == _SIM.bare_rf
        assert rf_e == _SIM.bare_rf


class TestMixedSignal:
    _FREQS = np.linspace(7.15, 7.25, 201)

    def test_p_e_zero_equals_ground_response(self) -> None:
        rf_g, rf_e = resonator_freqs(_SIM, flux=0.3)
        sig = mixed_signal(_SIM, self._FREQS, rf_g, rf_e, p_e=0.0)
        np.testing.assert_allclose(sig, s21(_SIM, self._FREQS, rf_g))

    def test_p_e_one_equals_excited_response(self) -> None:
        rf_g, rf_e = resonator_freqs(_SIM, flux=0.3)
        sig = mixed_signal(_SIM, self._FREQS, rf_g, rf_e, p_e=1.0)
        np.testing.assert_allclose(sig, s21(_SIM, self._FREQS, rf_e))

    def test_p_e_half_is_midpoint(self) -> None:
        rf_g, rf_e = resonator_freqs(_SIM, flux=0.3)
        sig_g = mixed_signal(_SIM, self._FREQS, rf_g, rf_e, p_e=0.0)
        sig_e = mixed_signal(_SIM, self._FREQS, rf_g, rf_e, p_e=1.0)
        sig_half = mixed_signal(_SIM, self._FREQS, rf_g, rf_e, p_e=0.5)
        np.testing.assert_allclose(sig_half, 0.5 * (sig_g + sig_e))

    def test_onetone_sweep_has_dip_near_rf_g(self) -> None:
        rf_g, rf_e = resonator_freqs(_SIM, flux=0.3)
        sig = mixed_signal(_SIM, self._FREQS, rf_g, rf_e, p_e=0.0)
        mag = np.abs(sig)
        dip_freq = self._FREQS[int(np.argmin(mag))]
        # Magnitude minimum (resonance dip) sits at the ground resonance.
        assert abs(dip_freq - rf_g) < 5e-3  # within 5 MHz of the swept grid
        # Off-resonant transmission recovers toward the unit baseline.
        assert mag[0] > mag.min()
        assert mag[-1] > mag.min()

    def test_output_is_complex128(self) -> None:
        rf_g, rf_e = resonator_freqs(_SIM, flux=0.3)
        sig = mixed_signal(_SIM, self._FREQS, rf_g, rf_e, p_e=0.5)
        assert sig.dtype == np.complex128


class TestDipDepthVsQi:
    """Verify that Qi controls the resonance dip depth via Qc derivation."""

    _FREQS = np.linspace(7.15, 7.25, 2001)

    def _dip_depth(self, Qi: float) -> float:
        """Return 1 - |S21_min| at p_e=0 (ground-state dip) for the given Qi."""
        sim = SimParams(
            EJ=8.5,
            EC=1.0,
            EL=0.5,
            flux_period=0.002,
            flux_half=0.001,
            T1=50.0,
            T2=30.0,
            T2_star=30.0,  # T2_star == T2 => gamma=0 (pure homogeneous)
            bare_rf=7.2,
            g=0.08,
            Ql=5000.0,
            Qi=Qi,
            snr=10.0,
            pi_gain_len=0.4,
        )
        rf_g, rf_e = resonator_freqs(sim, flux=0.3)
        sig = mixed_signal(sim, self._FREQS, rf_g, rf_e, p_e=0.0)
        return float(1.0 - np.abs(sig).min())

    def test_large_qi_gives_deep_dip(self) -> None:
        # Qi >> Ql: Qc ≈ Ql (coupling-limited); dip depth approaches 1 - Ql/Qi ≈ 1.
        dip = self._dip_depth(Qi=5_000_000.0)
        assert dip > 0.9, f"expected deep dip (>0.9) for large Qi, got {dip:.4f}"

    def test_small_qi_gives_shallow_dip(self) -> None:
        # Qi just above Ql: dip depth = 1 - Ql/Qi is small (lossy resonator).
        dip = self._dip_depth(Qi=5001.0)
        assert dip < 0.1, f"expected shallow dip (<0.1) for Qi≈Ql, got {dip:.4f}"

    def test_deeper_dip_for_larger_qi(self) -> None:
        # Monotonicity: larger Qi → deeper dip (less internal loss).
        dip_large = self._dip_depth(Qi=500_000.0)
        dip_small = self._dip_depth(Qi=6000.0)
        assert dip_large > dip_small, (
            f"expected dip_large ({dip_large:.4f}) > dip_small ({dip_small:.4f})"
        )


class TestFastFail:
    @pytest.mark.parametrize("bad_p_e", [-0.01, 1.01, 2.0, -1.0])
    def test_p_e_out_of_range_raises(self, bad_p_e: float) -> None:
        freqs = np.array([7.2], dtype=np.float64)
        rf_g, rf_e = resonator_freqs(_SIM, flux=0.3)
        with pytest.raises(ValueError, match=r"p_e must be in \[0, 1\]"):
            mixed_signal(_SIM, freqs, rf_g, rf_e, p_e=bad_p_e)
