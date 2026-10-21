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

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.spectra import (
    DEFAULT_GAP_EV as _DEF_DELTA,
)
from zcu_tools.simulate.spectra import (
    DEFAULT_IMPEDANCE_OHM as _DEF_Z,
)
from zcu_tools.simulate.spectra import (
    DEFAULT_MUTUAL_PHI0_PER_A as _DEF_M,
)
from zcu_tools.simulate.spectra import (
    DEFAULT_QP_DENSITY as _DEF_X_QP,
)
from zcu_tools.simulate.spectra import (
    capacitive_spectrum as _s_capacitive,
)
from zcu_tools.simulate.spectra import (
    charge_impedance_spectrum as _s_charge_impedance,
)
from zcu_tools.simulate.spectra import (
    flux_line_spectrum as _s_flux_bias_line,
)
from zcu_tools.simulate.spectra import (
    inductive_spectrum as _s_inductive,
)
from zcu_tools.simulate.spectra import (
    quasiparticle_spectrum as _s_quasiparticle,
)


class UnsupportedNoiseChannelError(ValueError):
    """A requested noise channel has no fast (numpy) implementation here."""


class UnsupportedNoiseOptionError(ValueError):
    """A noise-channel option (or top-level kwarg) the fast path does not support.

    The fast path fixes ``total=True`` and only accepts each channel's documented
    options, so an unknown option (e.g. ``total``) raises rather than being silently
    ignored — silently dropping it would return a wrong-but-plausible T1.
    """


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


# tolerated: Notebook fits tune noise/Temp separately from basis size and i/j.
def calculate_eff_t1_vs_flux_fast(  # noqa: PLR0913
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
    ``1 / total_rate`` in physical ns. Each spectral rate already includes its
    angular-frequency conversion (ADR-0071). ``params=(EJ, EC, EL)`` are GHz.
    ``fluxs`` must be a 1D array of shape ``(N,)`` in Φ/Φ₀, and ``Temp`` is K.
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

    out = np.where(total != 0.0, 1 / total, np.inf)
    return out.astype(np.float64, copy=False)


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
