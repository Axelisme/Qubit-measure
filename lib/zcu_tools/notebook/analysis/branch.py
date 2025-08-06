from typing import TYPE_CHECKING, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import trange

if TYPE_CHECKING:
    from scqubits import Fluxonium, HilbertSpace, Oscillator, ParameterSweep


def make_hilbertspace(
    params: Tuple[float, float, float],
    r_f: float,
    qub_dim: int,
    qub_cutoff: int,
    res_dim: int,
    g: float,
    flx: float = 0.5,
) -> "HilbertSpace":
    from scqubits import Fluxonium, HilbertSpace, Oscillator  # lazy import

    resonator = Oscillator(r_f, truncated_dim=res_dim)
    fluxonium = Fluxonium(*params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim)
    hilbertspace = HilbertSpace([fluxonium, resonator])
    hilbertspace.add_interaction(
        g=g, op1=fluxonium.n_operator, op2=resonator.creation_operator, add_hc=True
    )

    return hilbertspace


def make_bra_array(
    hilbertspace: "HilbertSpace", qub_dim: int, res_dim: int
) -> np.ndarray:
    return np.array(
        [
            np.sqrt(j) * hilbertspace.bare_productstate((j, m)).dag().full()
            for j in range(qub_dim)
            for m in range(res_dim)
        ]
    )


def calc_population(bra_array: np.ndarray, evec: np.ndarray) -> float:
    r"""
    Calculate the average population of the the given state

    Equation: P(i, n) = \sum_{j, m} j*|<j,m|\dash{i,n}>|^2
    """
    return np.sum(np.abs(np.dot(bra_array, evec)) ** 2)


def branch_population(
    hilbertspace: "HilbertSpace", branchs: np.ndarray, upto: int = -1
) -> np.ndarray:
    """
    Calculate the average population of the states in branchs upto provided photon number
    """
    fluxonium, resonator = hilbertspace.subsystem_list
    qub_dim = fluxonium.truncated_dim
    res_dim = resonator.truncated_dim

    if not hilbertspace.lookup_exists():
        hilbertspace.generate_lookup(ordering="LX")

    bra_array = make_bra_array(hilbertspace, qub_dim, res_dim)

    _, evecs = hilbertspace.eigensys(evals_count=qub_dim * res_dim)
    dressed_indices = [
        hilbertspace.dressed_index((b, n)) for b in branchs for n in range(upto)
    ]

    def _calc_population(dressed_idx: int) -> float:
        return calc_population(bra_array, evecs[dressed_idx].full())

    populations = np.array(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(_calc_population)(dressed_idx) for dressed_idx in dressed_indices
        )
    ).reshape(len(branchs), upto)

    return populations


def plot_branch_population(
    branchs: list[int],
    populations: np.ndarray,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()

    for b in branchs:
        pop_b = populations[b]
        if np.ptp(pop_b) > 1.0:
            color = None
            label = f"Branch {b}"
        else:
            color = "lightgrey"
            label = None
        ax.plot(populations[b], label=label, color=color)
    ax.legend(loc="upper left")
    ax.grid()

    return fig, ax


def branch_population_over_flux(
    flxs: np.ndarray,
    params: Tuple[float, float, float],
    r_f: float,
    qub_dim: int,
    qub_cutoff: int,
    res_dim: int,
    g: float,
    upto: int = -1,
    branchs: Optional[list[int]] = None,
    batch_size: int = 10,
) -> np.ndarray:
    from scqubits import (  # lazy import
        Fluxonium,
        HilbertSpace,
        Oscillator,
        ParameterSweep,
    )

    hilbertspace = make_hilbertspace(params, r_f, qub_dim, qub_cutoff, res_dim, g)
    fluxonium, resonator = hilbertspace.subsystem_list

    def update_hilbertspace(flx: float) -> None:
        fluxonium.flux = flx

    bra_array = make_bra_array(hilbertspace, qub_dim, res_dim)

    # batching to reduce memory usage
    populations_list = []
    for i in trange(0, len(flxs), batch_size, desc="Batch"):
        batched_flxs = flxs[i : i + batch_size]

        sweep = ParameterSweep(
            hilbertspace=hilbertspace,
            paramvals_by_name={"flux": batched_flxs},
            update_hilbertspace=update_hilbertspace,
            evals_count=qub_dim * res_dim,
            subsys_update_info={"flux": [fluxonium]},
            labeling_scheme="LX",
        )

        def _calc_branch_populations(
            paramsweep: ParameterSweep, paramindex_tuple: tuple, **kwargs
        ) -> np.ndarray:
            # (qub_dim * res_dim, (qub_dim * res_dim, 1))
            evecs = paramsweep["evecs"][paramindex_tuple]

            def _calc_population(b, n) -> float:
                dressed_idx = paramsweep.dressed_index((b, n), paramindex_tuple)
                return calc_population(bra_array, evecs[dressed_idx].full())

            populations = np.array(
                Parallel(n_jobs=-1, prefer="threads")(
                    delayed(_calc_population)(b, n)
                    for b in branchs
                    for n in range(upto)
                )
            ).reshape(len(branchs), upto)

            return populations

        sweep.add_sweep(_calc_branch_populations, sweep_name="branch_populations")

        populations_list.append(sweep["branch_populations"])

    populations = np.concatenate(populations_list)

    return populations
