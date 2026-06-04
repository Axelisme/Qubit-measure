from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TYPE_CHECKING, Any, Callable, Union

if TYPE_CHECKING:
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.param_sweep import ParameterSweep


@lru_cache(maxsize=32)
def _fluxonium_operators(
    params: tuple[float, float, float], qub_cutoff: int, qub_dim: int
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.complex128],
    float,
]:
    """The flux-independent fluxonium pieces of ``calculate_dispersive_vs_flux_fast``.

    Returns ``(lc_diag, cos_phi, sin_phi, n_op, EJ)`` — none depend on flux, so they
    are built ONCE per (params, cutoff, dim) and memoised. This is the live tuning's
    hot fix: ``cos_phi_operator`` / ``sin_phi_operator`` go through scipy's matrix
    ``cosm`` / ``sinm`` (an ``expm`` each), ~80 ms on a fresh ``Fluxonium`` — rebuilding
    them on every sample-flux drag / r_f move would dominate. With them cached, only
    the cheap per-flux recombination + eigensolve (~1 ms/flux) runs per update.

    The returned arrays are the cache's own copies — callers must NOT mutate them
    (the fast path only reads them).
    """
    from scqubits.core.fluxonium import Fluxonium

    fx = Fluxonium(*params, flux=0.0, cutoff=qub_cutoff, truncated_dim=qub_dim)
    dim = fx.hilbertdim()
    lc_diag = np.array(
        [(i + 0.5) * fx.plasma_energy() for i in range(dim)], dtype=np.float64
    )
    cos_phi = np.asarray(fx.cos_phi_operator(beta=0.0), dtype=np.float64)
    sin_phi = np.asarray(fx.sin_phi_operator(beta=0.0), dtype=np.float64)
    n_op = np.asarray(fx.n_operator(), dtype=np.complex128)
    return lc_diag, cos_phi, sin_phi, n_op, float(fx.EJ)


def calculate_dispersive(
    params: tuple[float, float, float], flux: float, bare_rf: float, g: float
) -> tuple[float, float]:
    """
    Calculate the dispersive shift of ground and excited state
    """

    resonator_dim = 10
    cutoff = 30
    evals_count = 10

    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.hilbert_space import HilbertSpace
    from scqubits.core.oscillator import Oscillator

    resonator = Oscillator(bare_rf, truncated_dim=resonator_dim)
    fluxonium = Fluxonium(*params, flux=flux, cutoff=cutoff, truncated_dim=evals_count)
    hilbertspace = HilbertSpace([resonator, fluxonium])
    hilbertspace.add_interaction(
        g=g, op1=resonator.creation_operator, op2=fluxonium.n_operator, add_hc=True
    )
    hilbertspace.generate_lookup(ordering="LX")

    idx_00 = hilbertspace.dressed_index((0, 0))
    idx_10 = hilbertspace.dressed_index((1, 0))
    idx_01 = hilbertspace.dressed_index((0, 1))
    idx_11 = hilbertspace.dressed_index((1, 1))
    max_idx = max(idx_00, idx_10, idx_01, idx_11)

    evals = hilbertspace.eigenvals(evals_count=max_idx + 1)
    rf_0 = evals[idx_10] - evals[idx_00]
    rf_1 = evals[idx_11] - evals[idx_01]

    return rf_0, rf_1


def calculate_dispersive_sweep(
    sweep_list: NDArray[np.float64],
    update_fn: Callable[[Fluxonium, Any], None],
    g: float,
    bare_rf: float,
    progress: bool = True,
    res_dim: int = 5,
    qub_cutoff: int = 30,
    qub_dim: int = 20,
    return_dim: int = 2,
) -> tuple[NDArray[np.float64], ...]:
    """
    Calculate the dispersive shift of ground and excited state vs. params of fluxonium
    """

    import scqubits.settings as scq_settings
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.hilbert_space import HilbertSpace
    from scqubits.core.oscillator import Oscillator
    from scqubits.core.param_sweep import ParameterSweep

    resonator = Oscillator(bare_rf, truncated_dim=res_dim)
    fluxonium = Fluxonium(
        *(1.0, 1.0, 1.0), flux=0.5, cutoff=qub_cutoff, truncated_dim=qub_dim
    )
    hilbertspace = HilbertSpace([resonator, fluxonium])
    hilbertspace.add_interaction(
        g=g, op1=resonator.creation_operator, op2=fluxonium.n_operator, add_hc=True
    )

    def update_hilbertspace(sweep_param: Any) -> None:
        update_fn(fluxonium, sweep_param)

    old = scq_settings.PROGRESSBAR_DISABLED
    scq_settings.PROGRESSBAR_DISABLED = not progress
    sweep = ParameterSweep(
        hilbertspace,
        {"params": sweep_list},
        update_hilbertspace=update_hilbertspace,
        evals_count=res_dim * qub_dim,
        subsys_update_info={"params": [fluxonium]},
        labeling_scheme="LX",
    )
    scq_settings.PROGRESSBAR_DISABLED = old

    evals = sweep["evals"].toarray()

    rf_list = []
    idxs = np.arange(len(sweep_list))
    for i in range(return_dim):
        idx_0i = sweep.dressed_index((0, i)).toarray()
        idx_1i = sweep.dressed_index((1, i)).toarray()
        rf_list.append(evals[idxs, idx_1i] - evals[idxs, idx_0i])
    return tuple(rf_list)


def calculate_dispersive_vs_flux(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    bare_rf: float,
    g: float,
    progress: bool = True,
    res_dim: int = 10,
    qub_cutoff: int = 30,
    qub_dim: int = 10,
    return_dim: int = 2,
) -> tuple[NDArray[np.float64], ...]:
    """
    Calculate the dispersive shift of ground and excited state vs. flux
    """

    def update_hilbertspace(fluxonium: Fluxonium, flux: float) -> None:
        fluxonium.flux = flux
        fluxonium.EJ = params[0]
        fluxonium.EC = params[1]
        fluxonium.EL = params[2]

    return calculate_dispersive_sweep(
        fluxs,
        update_hilbertspace,
        g,
        bare_rf,
        progress,
        res_dim,
        qub_cutoff,
        qub_dim,
        return_dim,
    )


def calculate_chi_sweep(
    sweep_list: Union[NDArray, list],
    update_fn: Callable[[Fluxonium, Any], None],
    g: float,
    bare_rf: float,
    progress: bool = True,
    resonator_dim: int = 5,
    cutoff: int = 30,
    evals_count: int = 20,
) -> NDArray[np.float64]:
    """
    Calculate the chi of ground and excited state vs. params of fluxonium
    """

    import scqubits.settings as scq_settings
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.hilbert_space import HilbertSpace
    from scqubits.core.oscillator import Oscillator
    from scqubits.core.param_sweep import ParameterSweep

    resonator = Oscillator(bare_rf, truncated_dim=resonator_dim)
    fluxonium = Fluxonium(
        *(1.0, 1.0, 1.0), flux=0.5, cutoff=cutoff, truncated_dim=evals_count
    )
    hilbertspace = HilbertSpace([fluxonium, resonator])
    hilbertspace.add_interaction(
        g=g, op1=fluxonium.n_operator, op2=resonator.creation_operator, add_hc=True
    )

    def update_hilbertspace(sweep_param: Any) -> None:
        update_fn(fluxonium, sweep_param)

    old = scq_settings.PROGRESSBAR_DISABLED
    scq_settings.PROGRESSBAR_DISABLED = not progress
    sweep = ParameterSweep(
        hilbertspace,
        {"params": np.asarray(sweep_list)},
        update_hilbertspace=update_hilbertspace,
        evals_count=resonator_dim * evals_count,
        subsys_update_info={"params": [fluxonium]},
        labeling_scheme="LX",
    )
    scq_settings.PROGRESSBAR_DISABLED = old

    return sweep["chi"]["subsys1":0, "subsys2":1]


def calculate_chi_vs_flux(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    bare_rf: float,
    g: float,
    progress: bool = True,
    res_dim: int = 5,
    qub_cutoff: int = 30,
    qub_dim: int = 20,
) -> NDArray[np.float64]:
    """
    Calculate the dispersive shift of ground and excited state vs. flux
    Returns:
        chi: NDArray[np.float64], shape: (len(fluxs), res_dim)
    """

    def update_hilbertspace(fluxonium: Fluxonium, flux: float) -> None:
        fluxonium.flux = flux
        fluxonium.EJ = params[0]
        fluxonium.EC = params[1]
        fluxonium.EL = params[2]

    return calculate_chi_sweep(
        fluxs,
        update_hilbertspace,
        g,
        bare_rf,
        progress,
        res_dim,
        qub_cutoff,
        qub_dim,
    )


class DressedLabelingError(RuntimeError):
    """Two bare product states mapped to the same dressed level (labeling ambiguous).

    Raised by ``calculate_dispersive_vs_flux_fast`` when the simple overlap-argmax
    dressed labeling is not a bijection at some flux — which happens when the
    coupling is too strong / the levels too dense for the (0/1, i) states to track
    cleanly. The caller should fall back to ``calculate_dispersive_vs_flux``
    (scqubits, robust labeling) for those parameters.
    """


def calculate_dispersive_vs_flux_fast(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    bare_rf: float,
    g: float,
    res_dim: int = 4,
    qub_dim: int = 15,
    qub_cutoff: int = 30,
    return_dim: int = 2,
) -> tuple[NDArray[np.float64], ...]:
    """A scqubits-free, ~9x-faster ``calculate_dispersive_vs_flux`` (numpy only).

    Computes the same ground/excited dispersive resonator frequencies vs flux as
    ``calculate_dispersive_vs_flux`` — matched to it to 0.00000 MHz across the
    avoided-crossing region (see tests) — but bypasses scqubits' ``ParameterSweep``:

    1. The flux-independent fluxonium operators (``cos(phi)`` / ``sin(phi)`` / the LC
       diagonal / ``n``) are taken from scqubits ONCE; per flux only a cheap
       ``cos(phi + beta)`` recombination + a ``hilbertdim`` ``eigh`` is done (the same
       trick ``calculate_energy_vs_flux`` uses for the bare spectrum).
    2. The composite Hamiltonian ``H_res ⊗ I + I ⊗ H_qub + g(a†⊗n + a⊗n†)`` is built
       and diagonalised in numpy on the (res_dim × qub_dim) tensor basis.
    3. Each needed dressed level is labelled by the bare product state ``|n_r, n_q⟩``
       it overlaps most (argmax) — for the low (0/1, i) states in the dispersive
       regime this is unambiguous; a uniqueness guard raises ``DressedLabelingError``
       if it ever collides, so a caller can fall back to the scqubits path.

    The flux grid is folded into [0, 0.5] and deduplicated (the spectrum is periodic
    and even), then the result is mapped back to the input order. Returns ``return_dim``
    arrays (rf_0, rf_1, ...) in GHz, one value per input flux — same shape/units as
    ``calculate_dispersive_vs_flux``.
    """
    from .energies import _fold_unique_fluxs

    folded, sort_idxs, uni_idxs = _fold_unique_fluxs(fluxs)

    # Flux-independent fluxonium pieces — built once per (params, cutoff, dim) and
    # memoised (the expensive cos/sin matrix functions would otherwise dominate the
    # live single-point tuning path; see _fluxonium_operators).
    lc_diag, cos_phi, sin_phi, n_op, EJ = _fluxonium_operators(
        params, qub_cutoff, qub_dim
    )

    # Resonator pieces (number diagonal + ladder operators).
    H_res = bare_rf * np.diag(np.arange(res_dim).astype(np.float64))
    a = np.diag(np.sqrt(np.arange(1, res_dim)), 1).astype(np.float64)
    adag = a.T
    I_res = np.eye(res_dim)

    # The bare product states we need to track: |n_r, n_q> for n_r in 0,1 and
    # n_q in 0..return_dim-1, as composite-basis index n_r*qub_dim + n_q.
    needed = [(nr, nq) for nr in (0, 1) for nq in range(return_dim)]

    out = [np.empty(len(folded), dtype=np.float64) for _ in range(return_dim)]
    for k, flux in enumerate(folded):
        beta = 2.0 * np.pi * flux
        Hq_full = np.diag(lc_diag) - EJ * (
            cos_phi * np.cos(beta) - sin_phi * np.sin(beta)
        )
        ev, evec = np.linalg.eigh(Hq_full)
        ev = ev[:qub_dim]
        evec = evec[:, :qub_dim]
        n_eig = evec.conj().T @ n_op @ evec  # n in the fluxonium eigenbasis

        H = (
            np.kron(H_res, np.eye(qub_dim))
            + np.kron(I_res, np.diag(ev))
            + g * (np.kron(adag, n_eig) + np.kron(a, n_eig.conj().T))
        )
        E, V = np.linalg.eigh(H)
        probs = np.abs(V) ** 2  # column j is dressed state j; row = bare index

        dressed = {}
        for nr, nq in needed:
            bare = nr * qub_dim + nq
            dressed[(nr, nq)] = int(np.argmax(probs[bare]))
        if len(set(dressed.values())) != len(dressed):
            raise DressedLabelingError(
                f"ambiguous dressed labeling at flux={flux:.4f} "
                f"(g={g}, res_dim={res_dim}); fall back to the scqubits path"
            )

        for i in range(return_dim):
            out[i][k] = E[dressed[(1, i)]] - E[dressed[(0, i)]]

    # Map folded-unique results back to the input flux order.
    result = []
    for i in range(return_dim):
        full = np.empty(len(folded), dtype=np.float64)
        full[sort_idxs] = out[i]
        result.append(full[uni_idxs])
    return tuple(result)
