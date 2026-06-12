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

The T1 noise formulas (the spectral densities + Fermi rate) are transcribed verbatim
from ``scqubits/core/noise.py`` and validated pointwise against scqubits to machine
precision (see ``tests/simulate/fluxonium/test_coherence_fast.py``). The
result matches ``calculate_eff_t1_vs_flux`` to ~1e-13 relative. The supported channels
are the five Fluxonium T1 channels; an unsupported channel name fast-fails.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import scipy.constants as const
import scipy.special as special
from numpy.typing import NDArray

# Physical constants (mirrors scqubits NOISE_PARAMS / CONSTANTS / units).
_HBAR = const.hbar
_KB = const.k
_E = const.e
_H = const.h
_R_K = _H / _E**2  # Klitzing constant R_k = h / e^2
_TO_STD = 1e9  # units.to_standard_units: GHz system units -> Hz

# scqubits NOISE_PARAMS defaults used by the channels.
_DEF_Z = 50.0  # R_0, characteristic impedance (Ohm)
_DEF_M = 400.0  # mutual inductance (Phi_0 / A)
_DEF_X_QP = 3e-6  # quasiparticle density
_DEF_DELTA = 3.4e-4  # superconducting gap (eV)


class UnsupportedNoiseChannelError(ValueError):
    """A requested noise channel has no fast (numpy) implementation here."""


class UnsupportedNoiseOptionError(ValueError):
    """A noise-channel option (or top-level kwarg) the fast path does not support.

    The fast path fixes ``total=True`` and only accepts each channel's documented
    options, so an unknown option (e.g. ``total``) raises rather than being silently
    ignored — silently dropping it would return a wrong-but-plausible T1.
    """


def _therm_ratio(omega: float, T: float, *, omega_in_std: bool = False) -> float:
    """``hbar * omega / (k_B * T)`` — scqubits ``calc_therm_ratio`` (omega in system
    rad units = 2*pi*GHz unless already standard)."""
    o = omega if omega_in_std else omega * _TO_STD
    return _HBAR * o / (_KB * T)


def _ev_to_hz(val: float) -> float:
    return val * _E / _H


# --- spectral densities S(omega, T), omega in system rad units (2*pi*GHz) ----------
# Each is a verbatim port of the corresponding scqubits ``spectral_density`` closure.


def _s_capacitive(omega, T, EC, Q_cap=None):
    therm = _therm_ratio(omega, T)
    if Q_cap is None:
        q = 1e6 * (2 * np.pi * 6e9 / np.abs(omega * _TO_STD)) ** 0.7
    elif callable(Q_cap):
        q = Q_cap(omega, T)
    else:
        q = Q_cap
    s = 2 * 8 * EC / q * (1 / np.tanh(0.5 * np.abs(therm))) / (1 + np.exp(-therm))
    return s * 2 * np.pi


def _real(value: Any) -> Any:
    """The real part of a (possibly complex) impedance/admittance value.

    Uses ``np.real`` so it works for both a scalar impedance ``Z`` and an array-valued
    admittance ``Y_qp`` (the vectorised-over-flux quasiparticle path passes a length-N
    omega, making ``y`` a length-N array)."""
    return np.real(value)


def _s_charge_impedance(omega, T, Z=_DEF_Z):
    z = Z(omega) if callable(Z) else Z
    Q_c = _R_K / (8 * np.pi * _real(z))
    therm = _therm_ratio(omega, T)
    return 2 * omega / Q_c * (1 / np.tanh(0.5 * therm)) / (1 + np.exp(-therm))


def _s_flux_bias_line(omega, T, M=_DEF_M, Z=_DEF_Z):
    z = Z(omega) if callable(Z) else Z
    therm = _therm_ratio(omega, T)
    s = (
        2
        * (2 * np.pi) ** 2
        * M**2
        * omega
        * _HBAR
        / _real(z)
        * (1 / np.tanh(0.5 * therm))
        / (1 + np.exp(-therm))
    )
    return s * _TO_STD**2  # 2 powers of frequency -> standard units


def _s_inductive(omega, T, EL, Q_ind=None):
    therm = _therm_ratio(omega, T)
    if Q_ind is None:
        tr = abs(therm)
        tr500 = _therm_ratio(2 * np.pi * 500e6, T, omega_in_std=True)
        q = (
            500e6
            * (special.kv(0, 0.5 * tr500) * np.sinh(0.5 * tr500))
            / (special.kv(0, 0.5 * tr) * np.sinh(0.5 * tr))
        )
    elif callable(Q_ind):
        q = Q_ind(omega, T)
    else:
        q = Q_ind
    s = 2 * EL / q * (1 / np.tanh(0.5 * np.abs(therm))) / (1 + np.exp(-therm))
    return s * 2 * np.pi


def _s_quasiparticle(omega, T, EJ, x_qp=_DEF_X_QP, Delta=_DEF_DELTA, Y_qp=None):
    therm = _therm_ratio(omega, T)
    if Y_qp is None:
        ao = abs(omega)
        delta_hz = _ev_to_hz(Delta)
        omega_hz = (ao * _TO_STD) / (2 * np.pi)
        ej_hz = EJ * _TO_STD
        tr = _therm_ratio(ao, T)
        y = (
            np.sqrt(2 / np.pi)
            * (8 / _R_K)
            * (ej_hz / delta_hz)
            * (2 * delta_hz / omega_hz) ** 1.5
            * x_qp
            * np.sqrt(0.5 * tr)
            * special.kv(0, 0.5 * abs(tr))
            * np.sinh(0.5 * tr)
        )
    elif callable(Y_qp):
        y = Y_qp(omega, T)
    else:
        y = Y_qp
    return (
        2
        * _HBAR
        * omega
        / _E**2
        * _real(y)
        * (1 / np.tanh(0.5 * therm))
        / (1 + np.exp(-therm))
    )


# A spectral density S(omega, T): omega may be a scalar or a length-N flux array (the
# vs-flux path evaluates all fluxes at once), so the return mirrors the input shape.
SpectralDensity = Callable[[Any, float], Any]

# Each channel: (operator key, allowed per-channel option keys, spectral-density
# builder). The operator key indexes the per-flux eigenbasis operators built in the
# sweep; the allowed-keys set is validated so an unknown / unsupported option (e.g.
# ``total``, which the fast path fixes to True) raises instead of being silently
# ignored. The option keys mirror the scqubits channel signatures.
_CHANNELS: dict[str, tuple[str, frozenset[str], Callable[..., SpectralDensity]]] = {
    "t1_capacitive": (
        "n",
        frozenset({"Q_cap"}),
        lambda EC, **o: lambda w, T: _s_capacitive(w, T, EC, o.get("Q_cap")),
    ),
    "t1_charge_impedance": (
        "n",
        frozenset({"Z"}),
        lambda **o: lambda w, T: _s_charge_impedance(w, T, o.get("Z", _DEF_Z)),
    ),
    "t1_inductive": (
        "phi",
        frozenset({"Q_ind"}),
        lambda EL, **o: lambda w, T: _s_inductive(w, T, EL, o.get("Q_ind")),
    ),
    "t1_flux_bias_line": (
        "dHdflux",
        frozenset({"M", "Z"}),
        lambda **o: (
            lambda w, T: _s_flux_bias_line(w, T, o.get("M", _DEF_M), o.get("Z", _DEF_Z))
        ),
    ),
    "t1_quasiparticle_tunneling": (
        "sinhalf",
        frozenset({"Y_qp", "x_qp", "Delta"}),
        lambda EJ, **o: (
            lambda w, T: _s_quasiparticle(
                w,
                T,
                EJ,
                o.get("x_qp", _DEF_X_QP),
                o.get("Delta", _DEF_DELTA),
                o.get("Y_qp"),
            )
        ),
    ),
}

NoiseChannel = str | tuple[str, dict[str, Any]]


def _resolve_channels(
    noise_channels: Sequence[NoiseChannel],
    EJ: float,
    EC: float,
    EL: float,
) -> list[tuple[str, SpectralDensity]]:
    """Resolve ``noise_channels`` (str or (str, opts)) to (op_key, spectral_density).

    Each channel's option keys are validated against its allowed set, so an unknown
    option (e.g. ``total``, ``i``/``j`` placed in a channel dict) raises rather than
    being silently dropped.
    """
    resolved = []
    for ch in noise_channels:
        name, opts = (ch[0], dict(ch[1])) if isinstance(ch, tuple) else (ch, {})
        if name not in _CHANNELS:
            raise UnsupportedNoiseChannelError(
                f"noise channel {name!r} is not supported by the fast path "
                f"(supported: {sorted(_CHANNELS)})"
            )
        op_key, allowed, builder = _CHANNELS[name]
        unknown = set(opts) - allowed
        if unknown:
            raise UnsupportedNoiseOptionError(
                f"noise channel {name!r} got unsupported option(s) "
                f"{sorted(unknown)} (the fast path fixes total=True and accepts only "
                f"{sorted(allowed)} here)"
            )
        # The qubit-parameter args each builder needs are passed by name; the
        # validated per-channel options (Q_cap, M, Z, x_qp, ...) flow through **opts.
        sfunc = builder(EJ=EJ, EC=EC, EL=EL, **opts)
        resolved.append((op_key, sfunc))
    return resolved


@lru_cache(maxsize=32)
def _t1_operators(
    params: tuple[float, float, float], cutoff: int, qub_dim: int
) -> tuple[
    NDArray[np.float64],  # lc_diag (dim,)
    NDArray[np.float64],  # cos_phi
    NDArray[np.float64],  # sin_phi
    NDArray[np.complex128],  # n_op
    NDArray[np.float64],  # phi_op
    NDArray[np.float64],  # cos_half (alpha=0.5)
    NDArray[np.float64],  # sin_half (alpha=0.5)
    float,  # EJ
]:
    """The flux-independent native operators for the fast T1 path, taken from scqubits
    ONCE per (params, cutoff, qub_dim) and memoised.

    Building these costs ~110 ms on a fresh ``Fluxonium`` (``cos_phi_operator`` /
    ``sin_phi_operator`` each go through scipy's matrix ``cosm`` / ``sinm`` = an
    ``expm``). The notebook caller (t1_curve/base.py) sweeps several ``noise_values``
    with the same ``params`` / ``cutoff`` / ``qub_dim``, so without the cache that fixed
    cost is paid per call; with it, only the cheap per-flux recombination + batched
    eigensolve runs each time. Mirrors ``dispersive._fluxonium_operators``.

    The returned arrays are the cache's own copies — callers must NOT mutate them
    (the fast path only reads them).
    """
    from scqubits.core.fluxonium import Fluxonium

    fx = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=qub_dim)
    dim = fx.hilbertdim()
    lc_diag = np.array(
        [(k + 0.5) * fx.plasma_energy() for k in range(dim)], dtype=np.float64
    )
    cos_phi = np.asarray(fx.cos_phi_operator(beta=0.0), dtype=np.float64)
    sin_phi = np.asarray(fx.sin_phi_operator(beta=0.0), dtype=np.float64)
    n_op = np.asarray(fx.n_operator(), dtype=np.complex128)
    phi_op = np.asarray(fx.phi_operator(), dtype=np.float64)
    cos_half = np.asarray(fx.cos_phi_operator(alpha=0.5, beta=0.0), dtype=np.float64)
    sin_half = np.asarray(fx.sin_phi_operator(alpha=0.5, beta=0.0), dtype=np.float64)
    return lc_diag, cos_phi, sin_phi, n_op, phi_op, cos_half, sin_half, float(fx.EJ)


def calculate_eff_t1_vs_flux_fast(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    noise_channels: Sequence[NoiseChannel],
    Temp: float,
    cutoff: int = 40,
    qub_dim: int = 20,
    i: int = 1,
    j: int = 0,
    **other_noise_options,
) -> NDArray[np.float64]:
    """Effective T1 (ns) vs flux for a Fluxonium — scqubits-free, ~60x faster.

    Matches ``calculate_eff_t1_vs_flux`` to ~1e-13 relative (see tests). Combines the
    requested ``noise_channels`` as ``1/T1_eff = sum_k 1/T1_k`` and returns
    ``2*pi / total_rate`` in ns (the ``2*pi`` is the ns/rad -> ns conversion the
    scqubits path also applies).

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

    # Flux-independent native operators — built once and memoised across calls that
    # share (params, cutoff, qub_dim) (see _t1_operators).
    lc_diag, cos_phi, sin_phi, n_op, phi_op, cos_half, sin_half, EJ_val = _t1_operators(
        params, cutoff, qub_dim
    )

    chans = _resolve_channels(noise_channels, EJ, EC, EL)
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
        s = sfunc(omega, Temp) + sfunc(-omega, Temp)  # (N,)
        total += matelem2[op_key] * np.real(s)

    out = np.where(total != 0.0, 2 * np.pi / total, np.inf)
    return out.astype(np.float64, copy=False)


def calculate_eff_t1_fast(
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
