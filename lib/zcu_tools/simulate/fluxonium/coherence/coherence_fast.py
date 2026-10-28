"""scqubits-free fast T1-vs-flux for Fluxonium (~60x).

``calculate_eff_t1_vs_flux`` (the scqubits path) is dominated by two costs that this
module removes:

1. **The eigensolve** — scqubits' ``get_spectrum_vs_paramvals`` diagonalises a fresh
   ``cutoff``-dim Hamiltonian per flux. Here the flux-independent ``cos(phi)`` /
   ``sin(phi)`` operators are taken from scqubits ONCE and the per-flux Hamiltonian is
   recombined as ``H(flux) = H_LC - EJ*(cos(phi)*cos(beta) - sin(phi)*sin(beta))``
   (= ``-EJ*cos(phi + beta)``, ``beta = 2*pi*flux``) — one numpy ``eigh`` per flux.

2. **The per-flux matrix sine** — ``t1_flux_bias_line`` and
   ``t1_quasiparticle_tunneling`` need ``dH/dflux`` and ``sin(phi/2 + pi*flux)``, which
   scqubits rebuilds with ``scipy.linalg.sinm`` (a 40x40 matrix sine) on every flux.
   Here they are recombined from the once-computed ``sin/cos(alpha*phi)`` via
   ``sin(alpha*phi + beta) = sin(alpha*phi)*cos(beta) + cos(alpha*phi)*sin(beta)`` — no
   ``sinm`` per flux.

The T1 noise spectra are provided by the shared numeric ``simulate.spectra``
owner and validated pointwise against scqubits to machine precision (see ``tests/simulate/fluxonium/test_coherence_fast.py``). The
result matches ``calculate_eff_t1_vs_flux`` to ~1e-13 relative. The supported channels
are the five Fluxonium T1 channels; an unsupported channel name fast-fails.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._fast_common import (
    NoiseChannel,
    SpectralDensity,
    UnsupportedNoiseChannelError,
    UnsupportedNoiseOptionError,
    resolve_channels,
    t1_operators,
    validate_temperature,
)


# tolerated: Notebook fits tune noise/Temp separately from basis size and i/j.
def calculate_eff_t1_vs_flux_fast(  # noqa: PLR0913
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    noise_channels: Sequence[NoiseChannel],
    Temp: float | NDArray[np.float64],
    cutoff: int = 40,
    qub_dim: int = 20,
    i: int = 1,
    j: int = 0,
    **other_noise_options,
) -> NDArray[np.float64]:
    """Effective T1 (ns) vs flux for a Fluxonium — scqubits-free, ~60x faster.

    Matches ``calculate_eff_t1_vs_flux`` to ~1e-13 relative (see tests). Combines the
    requested ``noise_channels`` as ``1/T1_eff = sum_k 1/T1_k`` and returns
    ``1 / total_rate`` in physical ns. Each spectral rate already includes its
    angular-frequency conversion (ADR-0071). ``params=(EJ, EC, EL)`` are GHz.
    ``fluxs`` must be a 1D array of shape ``(N,)`` in Φ/Φ₀. ``Temp`` is K,
    either a nonnegative finite scalar or a positive finite 1D array of shape
    ``(N,)``. Invalid temperatures raise ValueError before the eigensolve.
    Array temperatures preserve per-channel scalar/callable/array options;
    callable spectra receive the full omega grid and one scalar K at a time.
    The output has shape ``(N,)`` and preserves the input flux order.
    ``cutoff``/``qub_dim`` select oscillator basis/retained levels.

    ``noise_channels`` items are ``"t1_capacitive"`` etc., or ``("t1_inductive",
    {"Q_ind": 5e8})`` with per-channel options (Q_cap / Q_ind / Y_qp / M / Z / x_qp /
    Delta), same as scqubits. Only the five Fluxonium T1 channels are supported; an
    unknown channel raises ``UnsupportedNoiseChannelError``. ``i, j`` select the
    transition (default 1->0).

    The fast path fixes ``total=True``; any other top-level keyword (e.g.
    ``total=False``) raises ``UnsupportedNoiseOptionError`` rather than being silently
    dropped.
    """
    if other_noise_options:
        raise UnsupportedNoiseOptionError(
            f"unsupported keyword option(s) {sorted(other_noise_options)} for the fast "
            f"T1 path (it fixes total=True; only cutoff / qub_dim / i / j and "
            f"per-channel options are accepted)"
        )

    EJ, EC, EL = params
    fluxs = np.asarray(fluxs, dtype=np.float64)
    validate_temperature(Temp, fluxs.shape)

    # Flux-independent native operators — built once and memoised across calls that
    # share (params, cutoff, qub_dim) (see t1_operators).
    lc_diag, cos_phi, sin_phi, n_op, phi_op, cos_half, sin_half, EJ_val = t1_operators(
        params, cutoff, qub_dim
    )

    chans = resolve_channels(noise_channels, EJ, EC, EL)
    # Which flux-dependent operators are actually needed (skip building unused ones).
    need = {op for op, _ in chans}

    n = len(fluxs)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    # --- Batched per-flux eigensolve ------------------------------------------------
    # Stack H(flux) = H_LC - EJ*cos(phi + beta) for all fluxes into (N, dim, dim) and
    # diagonalise in one np.linalg.eigh call (it vmaps over the leading axis). beta has
    # shape (N,); broadcasting (N,1,1)*(dim,dim) builds the stack without a Python loop.
    beta = 2.0 * np.pi * fluxs  # (N,)
    cb = np.cos(beta)[:, None, None]  # (N,1,1)
    sb = np.sin(beta)[:, None, None]
    Hq = np.diag(lc_diag)[None] - EJ_val * (cos_phi[None] * cb - sin_phi[None] * sb)
    evals_full, evecs_full = np.linalg.eigh(Hq)  # (N,dim), (N,dim,dim)
    evals = evals_full[:, :qub_dim]  # (N, qub_dim)
    evecs = evecs_full[:, :, :qub_dim]  # (N, dim, qub_dim)
    evecs_h = np.conj(np.swapaxes(evecs, -1, -2))  # (N, qub_dim, dim)

    def _to_eigbasis(op: NDArray[Any]) -> NDArray[Any]:
        # evecs^H @ op @ evecs, batched over flux: (N,d,d)->(N,qub,qub).
        return evecs_h @ (op[None] @ evecs)

    # Operator matrices are complex or real depending on the channel; NDArray[Any]
    # is honest here since some ops (phi, dHdflux, sinhalf) are float64.
    ops: dict[str, NDArray[Any]] = {}
    if "n" in need:
        ops["n"] = _to_eigbasis(n_op)
    if "phi" in need:
        ops["phi"] = _to_eigbasis(phi_op)
    if "dHdflux" in need:
        # dH/dflux = -2*pi*EJ*sin(phi + beta) (recombined, no sinm), stacked over flux.
        dHdflux = -2 * np.pi * EJ_val * (sin_phi[None] * cb + cos_phi[None] * sb)
        ops["dHdflux"] = evecs_h @ (dHdflux @ evecs)
    if "sinhalf" in need:
        bh = 0.5 * beta  # sin(phi/2 + pi*flux)
        cbh = np.cos(bh)[:, None, None]
        sbh = np.sin(bh)[:, None, None]
        sinhalf = sin_half[None] * cbh + cos_half[None] * sbh
        ops["sinhalf"] = evecs_h @ (sinhalf @ evecs)

    # --- Per-channel Fermi rates, vectorised over flux -------------------------------
    # omega = 2*pi*(E_i - E_j) is now a length-N vector; the spectral densities are
    # written with numpy ops so they accept the array directly (same scalar math).
    omega = 2 * np.pi * (evals[:, i] - evals[:, j])  # (N,)
    matelem2 = {  # |<i|op|j>|^2 per flux, only for the ops actually requested
        key: np.abs(op[:, i, j]) ** 2 for key, op in ops.items()
    }
    total = np.zeros(n, dtype=np.float64)
    for op_key, sfunc in chans:
        total += matelem2[op_key] * _equilibrium_spectrum(sfunc, omega, Temp)

    out = np.where(total != 0.0, 1 / total, np.inf)
    return out.astype(np.float64, copy=False)


def _equilibrium_spectrum(
    spectrum: SpectralDensity,
    omega: NDArray[np.float64],
    temperature: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    if not isinstance(temperature, np.ndarray):
        return np.asarray(
            np.real(spectrum(omega, temperature) + spectrum(-omega, temperature)),
            dtype=np.float64,
        )
    rates = np.empty_like(omega)
    for value in np.unique(temperature):
        mask = temperature == value
        # Keep the full omega grid so per-channel array options stay row-aligned.
        evaluated = spectrum(omega, float(value)) + spectrum(-omega, float(value))
        rates[mask] = np.real(evaluated)[mask]
    return rates


# tolerated: Public scalar T1 preserves the sweep's basis and transition keywords.
def calculate_eff_t1_fast(  # noqa: PLR0913
    flux: float,
    params: tuple[float, float, float],
    noise_channels: Sequence[NoiseChannel],
    Temp: float,
    cutoff: int = 40,
    qub_dim: int = 20,
    i: int = 1,
    j: int = 0,
    **other_noise_options,
) -> float:
    """Single-flux effective T1 (ns) — the scalar form of
    ``calculate_eff_t1_vs_flux_fast`` (one flux point). Unsupported options raise the
    same ``UnsupportedNoiseOptionError`` as the vs-flux form."""
    out = calculate_eff_t1_vs_flux_fast(
        params,
        np.array([flux], dtype=np.float64),
        noise_channels,
        Temp,
        cutoff=cutoff,
        qub_dim=qub_dim,
        i=i,
        j=j,
        **other_noise_options,
    )
    return float(out[0])


# Optionally expose the per-flux Fluxonium for callers that already hold one (parity
# with the ``*_with`` variants). The fast path is self-contained, so this just builds
# the operators from the given params — kept minimal.
__all__ = [
    "calculate_eff_t1_vs_flux_fast",
    "calculate_eff_t1_fast",
    "UnsupportedNoiseChannelError",
    "UnsupportedNoiseOptionError",
]
