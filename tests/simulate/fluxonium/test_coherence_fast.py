"""Correctness tests for calculate_eff_t1_vs_flux_fast (the scqubits-free T1 path).

calculate_eff_t1_vs_flux_fast bypasses scqubits: it diagonalises the per-flux
Hamiltonian in numpy (cos/sin recombination), recombines the flux-dependent noise
operators (dH/dflux, sin(phi/2)) without a per-flux matrix sine, and evaluates the
ported noise spectral densities directly. These tests assert it matches the trusted
scqubits ``calculate_eff_t1_vs_flux`` to ~1e-12 relative across parameter sets, every
supported noise channel, custom per-channel options, and the single-flux form.

They call scqubits for the reference, so they are a touch slow but still seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import scqubits.settings as scq_settings
from zcu_tools.simulate.fluxonium import (
    UnsupportedNoiseChannelError,
    UnsupportedNoiseOptionError,
    calculate_eff_t1,
    calculate_eff_t1_fast,
    calculate_eff_t1_vs_flux,
    calculate_eff_t1_vs_flux_fast,
)

scq_settings.T1_DEFAULT_WARNING = False

PARAMS = [(4.0, 1.0, 0.5), (5.0, 1.2, 0.4), (3.0, 1.5, 0.3)]
CHANNELS = [
    "t1_capacitive",
    "t1_charge_impedance",
    "t1_inductive",
    "t1_flux_bias_line",
    "t1_quasiparticle_tunneling",
]
_TEMP = 0.05
# Stop just short of the half-flux point: at exactly flux=0.5 the qubit sits at the
# parity-symmetric sweet spot where some noise matrix elements (e.g. the
# quasiparticle sin(phi/2)) vanish, so T1 -> inf and the per-channel rate is a tiny
# residual where ref/fast can differ in relative terms while both mean "infinite".
_FLUXS = np.linspace(0.01, 0.49, 25)


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("channel", CHANNELS)
def test_fast_matches_scqubits_per_channel(params, channel):
    noise = [(channel, {})]
    ref = calculate_eff_t1_vs_flux(
        _FLUXS, noise, _TEMP, params, cutoff=40, evals_count=20
    )
    fast = calculate_eff_t1_vs_flux_fast(
        params, _FLUXS, noise, _TEMP, cutoff=40, qub_dim=20
    )
    # all channels match scqubits to machine precision (measured ~1e-13 relative)
    np.testing.assert_allclose(fast, ref, rtol=1e-9)


def test_fast_matches_scqubits_all_channels_combined():
    params = (4.0, 1.0, 0.5)
    noise = [(c, {}) for c in CHANNELS]
    ref = calculate_eff_t1_vs_flux(
        _FLUXS, noise, _TEMP, params, cutoff=40, evals_count=20
    )
    fast = calculate_eff_t1_vs_flux_fast(params, _FLUXS, noise, _TEMP)
    np.testing.assert_allclose(fast, ref, rtol=1e-9)


def test_fast_respects_per_channel_options():
    # Custom Q_cap / Q_ind / M / Z flow through the noise-channel tuples the same way
    # scqubits consumes them.
    params = (4.0, 1.0, 0.5)
    noise = [
        ("t1_capacitive", {"Q_cap": 1e6}),
        ("t1_inductive", {"Q_ind": 5e8}),
        ("t1_flux_bias_line", {"M": 500.0, "Z": 30.0}),
    ]
    ref = calculate_eff_t1_vs_flux(_FLUXS, noise, _TEMP, params)
    fast = calculate_eff_t1_vs_flux_fast(params, _FLUXS, noise, _TEMP)
    np.testing.assert_allclose(fast, ref, rtol=1e-9)


def test_fast_single_flux_matches_scqubits():
    params = (4.0, 1.0, 0.5)
    noise = [(c, {}) for c in CHANNELS]
    ref = calculate_eff_t1(0.3, noise, _TEMP, params)
    fast = calculate_eff_t1_fast(0.3, params, noise, _TEMP)
    assert fast == pytest.approx(ref, rel=1e-9)


def test_fast_accepts_bare_string_channels():
    # noise_channels items may be bare strings (no options dict), like scqubits.
    params = (4.0, 1.0, 0.5)
    channels = ["t1_capacitive"]
    ref = calculate_eff_t1_vs_flux(_FLUXS, channels, _TEMP, params)  # type: ignore[arg-type]
    fast = calculate_eff_t1_vs_flux_fast(params, _FLUXS, channels, _TEMP)
    np.testing.assert_allclose(fast, ref, rtol=1e-9)


def test_qp_rate_vanishes_at_half_flux():
    # At flux=0.5 the quasiparticle sin(phi/2) matrix element vanishes by parity, so
    # both paths report an effectively-infinite T1 (a huge number) — they agree on the
    # physics (rate ~ 0) even though their tiny residual rates differ in relative terms.
    params = (4.0, 1.0, 0.5)
    fluxs = np.array([0.5])
    ref = calculate_eff_t1_vs_flux(
        fluxs, [("t1_quasiparticle_tunneling", {})], _TEMP, params
    )
    fast = calculate_eff_t1_vs_flux_fast(
        params, fluxs, [("t1_quasiparticle_tunneling", {})], _TEMP
    )
    assert ref[0] > 1e20 and fast[0] > 1e20  # both "infinite"


def test_unsupported_channel_raises():
    with pytest.raises(UnsupportedNoiseChannelError):
        calculate_eff_t1_vs_flux_fast(
            (4.0, 1.0, 0.5), _FLUXS, [("tphi_1_over_f_flux", {})], _TEMP
        )
    assert issubclass(UnsupportedNoiseChannelError, ValueError)


_PARAMS = (4.0, 1.0, 0.5)


def test_top_level_total_false_raises():
    # The fast path fixes total=True; total=False must error, not be silently dropped.
    with pytest.raises(UnsupportedNoiseOptionError):
        calculate_eff_t1_vs_flux_fast(
            _PARAMS, _FLUXS, [("t1_capacitive", {})], _TEMP, total=False
        )
    assert issubclass(UnsupportedNoiseOptionError, ValueError)


def test_single_flux_total_false_raises():
    with pytest.raises(UnsupportedNoiseOptionError):
        calculate_eff_t1_fast(0.3, _PARAMS, [("t1_capacitive", {})], _TEMP, total=False)


def test_per_channel_total_false_raises():
    with pytest.raises(UnsupportedNoiseOptionError):
        calculate_eff_t1_vs_flux_fast(
            _PARAMS, _FLUXS, [("t1_capacitive", {"total": False})], _TEMP
        )


def test_per_channel_unknown_option_raises():
    # any option a channel does not accept is rejected (not silently ignored)
    with pytest.raises(UnsupportedNoiseOptionError):
        calculate_eff_t1_vs_flux_fast(
            _PARAMS, _FLUXS, [("t1_capacitive", {"foo": 1})], _TEMP
        )
    # an option valid for a different channel (M is for flux_bias_line) is also wrong
    with pytest.raises(UnsupportedNoiseOptionError):
        calculate_eff_t1_vs_flux_fast(
            _PARAMS, _FLUXS, [("t1_capacitive", {"M": 400.0})], _TEMP
        )


def test_valid_per_channel_options_still_work():
    # the validation must not break the legitimate option keys
    noise = [
        ("t1_capacitive", {"Q_cap": 1e6}),
        ("t1_inductive", {"Q_ind": 5e8}),
        ("t1_flux_bias_line", {"M": 500.0, "Z": 30.0}),
        ("t1_quasiparticle_tunneling", {"x_qp": 1e-6, "Delta": 3e-4}),
    ]
    out = calculate_eff_t1_vs_flux_fast(_PARAMS, _FLUXS, noise, _TEMP)
    assert np.all(np.isfinite(out))
