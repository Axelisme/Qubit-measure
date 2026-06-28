"""Tests for sim/readout.py — the dispersive readout (physics -> IQ) model.

Covers:
- resonator_freqs: dressed rf_g/rf_e near bare_rf with a non-zero dispersive shift.
- Q3 fallback: DressedLabelingError degrades to (bare_rf, bare_rf) + warning.
- mixed_signal: takes the (rf_g, rf_e) dressed frequencies directly (the engine
  resolves them once); p_e endpoints match S21(rf_g)/S21(rf_e), midpoint is the
  mean, and an onetone sweep shows a resonance dip near rf_g.
- Fast-fail: p_e outside [0, 1] raises.
- envelope_at: const window 1/0, gauss peak-at-center, flat_top flat region, zero
  before the pulse start.
- decimated_trace (model A): const readout -> trig_offset-delayed square of the
  steady mixed S21; p_e endpoints/midpoint; output length == ts length; 1-D guard.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.waveform import (
    ArbWaveformCfg,
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
from zcu_tools.program.v2.sim import readout
from zcu_tools.program.v2.sim.readout import (
    decimated_trace,
    effective_signal_samples,
    mixed_signal,
    noise_std_sample_scale,
    readout_drive_amplitude,
    resonator_freqs,
    s21,
)
from zcu_tools.program.v2.sim.waveforms import envelope_at
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

    def test_same_flux_reuses_cached_prediction(self, monkeypatch) -> None:
        readout._cached_resonator_freqs.cache_clear()
        calls = 0

        def fake_calculate(params, fluxes, bare_rf, g):
            nonlocal calls
            calls += 1
            flux = float(fluxes[0])
            return (
                np.array([bare_rf - 1e-3 * flux], dtype=np.float64),
                np.array([bare_rf + 1e-3 * flux], dtype=np.float64),
            )

        monkeypatch.setattr(
            readout, "calculate_dispersive_vs_flux_fast", fake_calculate
        )
        try:
            first = resonator_freqs(_SIM, flux=0.3)
            second = resonator_freqs(_SIM, flux=0.3)
            assert second == first
            assert calls == 1

            third = resonator_freqs(_SIM, flux=0.31)
            assert third != first
            assert calls == 2
        finally:
            readout._cached_resonator_freqs.cache_clear()


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


class TestIntegrationHelpers:
    """Pure helpers for gain, signal area, and integrated noise scaling."""

    def test_readout_drive_amplitude_defaults_direct_readout_to_unity(self) -> None:
        assert readout_drive_amplitude(None) == pytest.approx(1.0)

    @pytest.mark.parametrize("gain", [0.0, 0.05, 0.5, -0.25])
    def test_readout_drive_amplitude_returns_finite_gain(self, gain: float) -> None:
        assert readout_drive_amplitude(gain) == pytest.approx(gain)

    @pytest.mark.parametrize("gain", [float("nan"), float("inf"), -float("inf")])
    def test_readout_drive_amplitude_rejects_nonfinite_gain(self, gain: float) -> None:
        with pytest.raises(ValueError, match="readout gain must be finite"):
            readout_drive_amplitude(gain)

    def test_effective_signal_samples_direct_readout_counts_window(self) -> None:
        ts = np.linspace(0.0, 1.0, 11, endpoint=False, dtype=np.float64)
        assert effective_signal_samples(None, None, ts) == pytest.approx(11.0)

    def test_effective_signal_samples_const_full_window(self) -> None:
        cfg = _const_readout_cfg(length=1.0)
        ts = np.linspace(0.0, 1.0, 11, endpoint=False, dtype=np.float64)
        assert effective_signal_samples(cfg, 1.0, ts) == pytest.approx(11.0)

    def test_effective_signal_samples_const_shorter_than_window(self) -> None:
        cfg = _const_readout_cfg(length=0.5)
        ts = np.linspace(0.0, 1.0, 10, endpoint=False, dtype=np.float64)
        assert effective_signal_samples(cfg, 0.5, ts) == pytest.approx(5.0)

    def test_effective_signal_samples_shaped_pulse_is_finite_and_smaller(self) -> None:
        cfg = _gauss_readout_cfg(length=1.0, sigma=0.2)
        ts = np.linspace(0.0, 1.0, 101, endpoint=False, dtype=np.float64)
        area = effective_signal_samples(cfg, 1.0, ts)
        assert 0.0 < area < ts.size

    @pytest.mark.parametrize(
        "sample_times",
        [
            np.zeros((2, 3), dtype=np.float64),
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([], dtype=np.float64),
        ],
    )
    def test_effective_signal_samples_rejects_bad_sample_axis(
        self, sample_times: NDArray[np.float64]
    ) -> None:
        with pytest.raises(ValueError):
            effective_signal_samples(None, None, sample_times)

    @pytest.mark.parametrize("pulse_length", [0.0, -1.0, float("nan")])
    def test_effective_signal_samples_rejects_bad_pulse_length(
        self, pulse_length: float
    ) -> None:
        cfg = _const_readout_cfg(length=1.0)
        ts = np.array([0.0], dtype=np.float64)
        with pytest.raises(ValueError, match="pulse_length_us"):
            effective_signal_samples(cfg, pulse_length, ts)

    def test_noise_std_sample_scale_is_sqrt_sample_count(self) -> None:
        assert noise_std_sample_scale(1) == pytest.approx(1.0)
        assert noise_std_sample_scale(16) == pytest.approx(4.0)

    @pytest.mark.parametrize("n_samples", [0, -1, 1.5])
    def test_noise_std_sample_scale_rejects_bad_sample_count(
        self, n_samples: int
    ) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            noise_std_sample_scale(n_samples)


def _const_readout_cfg(length: float = 1.0) -> PulseCfg:
    """A const-waveform readout pulse cfg (peak envelope 1 over its length)."""
    return PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=0,
        nqz=1,
        freq=7200.0,
        gain=1.0,
    )


def _gauss_readout_cfg(length: float = 1.0, sigma: float = 0.25) -> PulseCfg:
    return PulseCfg(
        waveform=GaussWaveformCfg(length=length, sigma=sigma),
        ch=0,
        nqz=1,
        freq=7200.0,
        gain=1.0,
    )


def _flat_top_readout_cfg(length: float = 1.0, ramp: float = 0.2) -> PulseCfg:
    return PulseCfg(
        waveform=FlatTopWaveformCfg(
            length=length,
            raise_waveform=GaussWaveformCfg(length=ramp, sigma=ramp / 4.0),
        ),
        ch=0,
        nqz=1,
        freq=7200.0,
        gain=1.0,
    )


class TestEnvelopeAt:
    """envelope_at: peak-normalized shape, zero outside the pulse window."""

    def test_const_is_one_in_window_zero_outside(self) -> None:
        cfg = _const_readout_cfg(length=1.0)
        t = np.array([-0.5, 0.0, 0.5, 0.999, 1.0, 1.5], dtype=np.float64)
        amp = envelope_at(cfg, t, length=1.0)
        # In window [0, 1): exactly 1; outside: 0.
        np.testing.assert_array_equal(amp, [0.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    def test_gauss_peaks_at_center(self) -> None:
        cfg = _gauss_readout_cfg(length=1.0, sigma=0.25)
        t = np.linspace(0.0, 1.0, 101, dtype=np.float64)
        amp = envelope_at(cfg, t, length=1.0)
        # Peak (≈1) sits at the pulse center.
        peak_idx = int(np.argmax(amp))
        assert abs(t[peak_idx] - 0.5) < 0.02
        assert abs(amp[peak_idx] - 1.0) < 1e-6
        # Outside the window the envelope is 0.
        assert envelope_at(cfg, np.array([-0.1, 1.1]), length=1.0).tolist() == [
            0.0,
            0.0,
        ]

    def test_flat_top_flat_region_is_one(self) -> None:
        cfg = _flat_top_readout_cfg(length=1.0, ramp=0.2)
        # Flat top spans [ramp/2, length - ramp/2) = [0.1, 0.9).
        t_flat = np.array([0.1, 0.3, 0.5, 0.7, 0.89], dtype=np.float64)
        amp_flat = envelope_at(cfg, t_flat, length=1.0)
        np.testing.assert_allclose(amp_flat, 1.0)
        # Mid-rise (t = ramp/4 = 0.05) is strictly between 0 and 1.
        amp_rise = envelope_at(cfg, np.array([0.05]), length=1.0)[0]
        assert 0.0 < amp_rise < 1.0
        # Symmetric falling ramp matches the rising ramp by mirror.
        amp_fall = envelope_at(cfg, np.array([1.0 - 0.05]), length=1.0)[0]
        assert abs(amp_fall - amp_rise) < 1e-9

    def test_arb_waveform_samples_iq_magnitude_on_reference_axis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_get(name: str):
            assert name == "arb_readout"
            return (
                np.array([0.0, 0.3, 0.6], dtype=np.float64),
                np.array([0.0, 0.4, 0.8], dtype=np.float64),
                np.array([0.0, 0.5, 1.0], dtype=np.float64),
            )

        monkeypatch.setattr(
            "zcu_tools.meta_tool.arb_waveform.ArbWaveformDatabase.get", fake_get
        )

        cfg = PulseCfg(
            waveform=ArbWaveformCfg(data="arb_readout"),
            ch=0,
            nqz=1,
            freq=7200.0,
            gain=1.0,
        )
        t = np.array([-0.1, 0.0, 0.5, 0.75, 1.0], dtype=np.float64)
        amp = envelope_at(cfg, t, length=1.0)

        np.testing.assert_allclose(amp, [0.0, 0.0, 0.5, 0.75, 0.0])

    def test_unknown_window_is_zero_before_start(self) -> None:
        cfg = _const_readout_cfg(length=1.0)
        # Everything before the pulse start is 0 (trig_offset region in the trace).
        amp = envelope_at(cfg, np.array([-1.0, -0.01]), length=1.0)
        np.testing.assert_array_equal(amp, [0.0, 0.0])


class TestDecimatedTrace:
    """decimated_trace: const readout square pulse × steady mixed S21.

    The envelope is shifted by ``_SIM.timeFly`` (the readout time of flight, 0.5 µs
    here): the trace is ~0 for program-time ``ts < timeFly`` and the readout pulse
    appears in ``[timeFly, timeFly + pulse_length)``.
    """

    def _rf(self) -> tuple[float, float]:
        return resonator_freqs(_SIM, flux=0.3)

    def test_const_trace_is_timefly_shifted_square_of_steady_s21(self) -> None:
        rf_g, rf_e = self._rf()
        f_ro = rf_g  # probe on the ground resonance
        tof = _SIM.timeFly
        ro_len = 1.0
        cfg = _const_readout_cfg(length=ro_len)
        # Program-time axis: 0 .. timeFly + ro_len (envelope sits at [timeFly, ...)).
        ts = np.linspace(0.0, tof + ro_len, 200, dtype=np.float64)
        trace = decimated_trace(_SIM, ts, cfg, ro_len, f_ro, rf_g, rf_e, p_e=0.0)
        assert trace.shape == ts.shape
        assert trace.dtype == np.complex128

        steady = s21(_SIM, np.array([f_ro]), rf_g)[0]
        before = ts < tof
        inside = (ts >= tof) & (ts < tof + ro_len)
        # Before timeFly: ~0 (signal not yet received); inside: steady S21 point.
        np.testing.assert_allclose(trace[before], 0.0)
        np.testing.assert_allclose(trace[inside], steady)

    def test_trace_window_uses_pulse_length(self) -> None:
        rf_g, rf_e = self._rf()
        f_ro = rf_g
        tof = _SIM.timeFly
        pulse_len = 0.5
        cfg = _const_readout_cfg(length=pulse_len)
        ts = np.array([tof + 0.1, tof + pulse_len + 0.1], dtype=np.float64)

        trace = decimated_trace(
            _SIM,
            ts,
            cfg,
            pulse_len,
            f_ro,
            rf_g,
            rf_e,
            p_e=0.0,
        )

        steady = s21(_SIM, np.array([f_ro]), rf_g)[0]
        assert trace[0] == pytest.approx(steady)
        assert trace[1] == pytest.approx(0.0)

    def test_p_e_endpoints_and_midpoint(self) -> None:
        rf_g, rf_e = self._rf()
        f_ro = rf_g
        tof = _SIM.timeFly
        ro_len = 1.0
        cfg = _const_readout_cfg(length=ro_len)
        # Sample strictly inside the (timeFly-shifted) window so every point is in
        # the flat steady region.
        ts = np.linspace(tof, tof + ro_len, 50, endpoint=False, dtype=np.float64)

        s_g = s21(_SIM, np.array([f_ro]), rf_g)[0]
        s_e = s21(_SIM, np.array([f_ro]), rf_e)[0]

        tr_g = decimated_trace(_SIM, ts, cfg, ro_len, f_ro, rf_g, rf_e, p_e=0.0)
        tr_e = decimated_trace(_SIM, ts, cfg, ro_len, f_ro, rf_g, rf_e, p_e=1.0)
        tr_h = decimated_trace(_SIM, ts, cfg, ro_len, f_ro, rf_g, rf_e, p_e=0.5)
        # p_e=0 -> S21(rf_g), p_e=1 -> S21(rf_e), p_e=0.5 -> linear midpoint.
        np.testing.assert_allclose(tr_g, s_g)
        np.testing.assert_allclose(tr_e, s_e)
        np.testing.assert_allclose(tr_h, 0.5 * (s_g + s_e))

    def test_length_matches_ts(self) -> None:
        rf_g, rf_e = self._rf()
        cfg = _const_readout_cfg(length=1.0)
        for n in (1, 7, 64):
            ts = np.linspace(0.0, _SIM.timeFly + 1.0, n, dtype=np.float64)
            trace = decimated_trace(_SIM, ts, cfg, 1.0, rf_g, rf_g, rf_e, p_e=0.0)
            assert trace.shape == (n,)

    def test_non_1d_ts_raises(self) -> None:
        rf_g, rf_e = self._rf()
        cfg = _const_readout_cfg(length=1.0)
        ts = np.zeros((2, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="ts must be a 1-D array"):
            decimated_trace(_SIM, ts, cfg, 1.0, rf_g, rf_g, rf_e, p_e=0.0)
