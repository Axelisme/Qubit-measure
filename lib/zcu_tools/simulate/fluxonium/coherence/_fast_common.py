"""Shared spectral admission and native-operator cache for both T1 forwards."""

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, NamedTuple

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

    Each forward selects its rate directions internally and accepts only the
    documented spectral options. Unknown options (including ``total``) raise
    instead of silently changing the physical meaning of a lifetime.
    """


# A spectral density S(omega, T): omega may be a scalar or a length-N flux array (the
# vs-flux path evaluates all fluxes at once), so the return mirrors the input shape.
SpectralDensity = Callable[[Any, float], Any]

# Each channel: (operator key, allowed per-channel option keys, spectral-density
# builder). The operator key indexes the per-flux eigenbasis operators built in the
# sweep; the allowed-keys set is validated so an unknown / unsupported option (e.g.
# ``total``, which each forward selects internally) raises instead of being
# silently ignored. The option keys mirror the scqubits channel signatures.
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


def resolve_channels(
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
                f"{sorted(unknown)} (the forward selects directions internally and "
                f"accepts only {sorted(allowed)} here)"
            )
        # The qubit-parameter args each builder needs are passed by name; the
        # validated per-channel options (Q_cap, M, Z, x_qp, ...) flow through **opts.
        sfunc = builder(EJ=EJ, EC=EC, EL=EL, **opts)
        resolved.append((op_key, sfunc))
    return resolved


class NativeT1Operators(NamedTuple):
    """Cached oscillator-basis operators; callers must not mutate the arrays.

    lc_diag: LC energies/h, shape (cutoff,), GHz. cos_phi/sin_phi: cos(phi) and
    sin(phi), shape (cutoff,cutoff). n_op: dimensionless Cooper-pair charge.
    phi_op: phase in radians. cos_half/sin_half: cos(phi/2) and sin(phi/2).
    All operators have shape (cutoff,cutoff). ej: Josephson energy/h in GHz.
    """

    lc_diag: NDArray[np.float64]
    cos_phi: NDArray[np.float64]
    sin_phi: NDArray[np.float64]
    n_op: NDArray[np.complex128]
    phi_op: NDArray[np.float64]
    cos_half: NDArray[np.float64]
    sin_half: NDArray[np.float64]
    ej: float


@lru_cache(maxsize=32)
def t1_operators(
    params: tuple[float, float, float], cutoff: int, qub_dim: int
) -> NativeT1Operators:
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
    return NativeT1Operators(
        lc_diag, cos_phi, sin_phi, n_op, phi_op, cos_half, sin_half, float(fx.EJ)
    )


def validate_temperature(
    temperature: float | NDArray[np.float64], shape: tuple[int, ...]
) -> None:
    """Require finite scalar K >=0 or positive 1D array matching shape.

    Raise ValueError for invalid temperatures; inputs are not changed.
    """
    if isinstance(temperature, np.ndarray):
        if temperature.ndim != 1 or temperature.shape != shape:
            raise ValueError("Temp array must be 1D with the same shape as fluxs")
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
            raise ValueError("Temp array must be finite and positive")
    elif not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Temp scalar must be finite and nonnegative")
