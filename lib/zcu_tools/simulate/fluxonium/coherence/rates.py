"""Typed multilevel loss inputs, using the existing scqubits coherence seam."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..operators import validate_circuit
from .coherence import calculate_eff_t1_with


@dataclass(frozen=True)
class FluxoniumNoise:
    """Independent losses and effective echo flux noise for one circuit.

    q_cap/q_ind: finite positive constant capacitive/inductive Q; None disables
    the channel. xqp: finite nonnegative quasiparticle fraction, default zero
    (disabled), using the scqubits default superconducting gap 3.4e-4 eV.
    aphi: finite nonnegative Phi0/sqrt(Hz) at 1 Hz; used by the model's effective
    Markov echo dephasing, not a full time-domain 1/f bath. Invalid values raise
    ValueError. All channels are disabled by default.
    """

    q_cap: float | None = None
    q_ind: float | None = None
    xqp: float = 0.0
    aphi: float = 0.0

    def __post_init__(self) -> None:
        for quality in (self.q_cap, self.q_ind):
            if quality is not None and (not np.isfinite(quality) or quality <= 0):
                raise ValueError("loss Q must be finite positive or None")
        if not np.isfinite([self.xqp, self.aphi]).all() or min(self.xqp, self.aphi) < 0:
            raise ValueError("xqp and aphi must be finite nonnegative")


def calculate_directional_loss_rates(
    params: tuple[float, float, float],
    flux: float,
    esys: tuple[NDArray[np.float64], NDArray[np.complex128]],
    noise: FluxoniumNoise,
    *,
    cutoff: int,
    temperature_k: float,
) -> NDArray[np.float64]:
    """Return physical directional rates R[target,source], ns^-1, diagonal zero.

    params=(EJ,EC,EL) GHz and cutoff follow calculate_oscillator_operators.
    flux is finite Phi/Phi0; esys contains ascending energies (M,) GHz and
    orthonormal native oscillator columns (cutoff,M). Projected eigenpairs are
    accepted: the caller owns their truncation accuracy. noise supplies constant
    Q and quasiparticle channels; aphi is handled separately by model dephasing.
    temperature_k is finite nonnegative K. Each pair uses the existing effective
    T1 helper with total=False, not an equilibrium lifetime as a one-way rate.
    Invalid shapes/values raise ValueError; scqubits errors propagate. Does not
    mutate eigenpairs or global settings. Empty loss channels give a zero matrix.
    """
    from ._loss_circuit import DirectionalLossCircuit

    validate_circuit(params, cutoff)
    energy, vectors = esys
    if (
        not np.isfinite(flux)
        or not np.isfinite(temperature_k)
        or temperature_k < 0
        or energy.ndim != 1
        or not len(energy)
        or not np.isfinite(energy).all()
        or np.any(np.diff(energy) <= 0)
        or vectors.shape != (cutoff, len(energy))
        or not np.isfinite(vectors).all()
        or not np.allclose(
            vectors.conj().T @ vectors, np.eye(len(energy)), atol=1e-10, rtol=0
        )
    ):
        raise ValueError(
            "loss rates require finite flux, temperature and ordered native eigenpairs"
        )
    rates = np.zeros((len(energy), len(energy)))
    # This circuit dies with this call. Only native operators are reused; scalar
    # matrix elements and directional T1 evaluation retain scqubits' arithmetic.
    circuit = DirectionalLossCircuit(
        *params, flux=flux, cutoff=cutoff, truncated_dim=len(energy)
    )
    channels = (
        ("t1_capacitive", "Q_cap", noise.q_cap),
        ("t1_inductive", "Q_ind", noise.q_ind),
        ("t1_quasiparticle_tunneling", "x_qp", noise.xqp or None),
    )
    for name, option, strength in channels:
        if strength is None:
            continue
        for source in range(len(energy)):
            for target in range(len(energy)):
                if source == target:
                    continue
                lifetime = calculate_eff_t1_with(
                    flux,
                    [(name, {option: strength})],
                    temperature_k,
                    circuit,
                    esys,
                    transition=(source, target),
                    total=False,
                )
                rates[target, source] += 1 / lifetime
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("loss channels produced invalid directional rates")
    return rates
