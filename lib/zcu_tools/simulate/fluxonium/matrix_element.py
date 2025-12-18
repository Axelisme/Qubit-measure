from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep
    from scqubits.core.storage import SpectrumData


def calculate_n_oper(
    params: Tuple[float, float, float],
    flx: float,
    return_dim: int = 4,
    esys: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
) -> NDArray[np.float64]:
    """
    Calculate the matrix elements of the fluxonium
    """

    cutoff = 30

    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(*params, flux=flx, cutoff=cutoff, truncated_dim=return_dim)
    if esys is None:
        esys = fluxonium.eigensys(evals_count=return_dim)

    matrix_elements = fluxonium.n_operator(energy_esys=esys)

    return matrix_elements[:return_dim, :return_dim]


def calculate_n_oper_vs_flx(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    return_dim: int = 4,
    spectrum_data: Optional[SpectrumData] = None,
) -> Tuple[SpectrumData, NDArray[np.float64]]:
    """
    Calculate the matrix elements of the fluxonium vs. a parameter
    """

    if spectrum_data is None:
        cutoff = 40

        from scqubits.core.fluxonium import Fluxonium

        fluxonium = Fluxonium(
            *params, flux=0.5, cutoff=cutoff, truncated_dim=return_dim
        )
        spectrum_data = fluxonium.get_matelements_vs_paramvals(
            operator="n_operator",
            param_name="flux",
            param_vals=flxs,
            evals_count=return_dim,
        )
    matrix_elements = spectrum_data.matrixelem_table

    return spectrum_data, matrix_elements[:, :return_dim, :return_dim]


def calculate_system_n_oper_vs_flx(
    params: Tuple[float, float, float],
    flxs: NDArray[np.float64],
    r_f: float,
    g: float,
    return_dim: int = 4,
    progress: bool = True,
    sweep: Optional[ParameterSweep] = None,
) -> Tuple[ParameterSweep, NDArray[np.float64]]:
    """
    Calculate the matrix elements of the system over a parameter sweep
    """

    if sweep is None:
        cutoff = 50
        qub_dim = 20
        res_dim = 10

        import scqubits.settings as scq_settings
        from scqubits.core.fluxonium import Fluxonium
        from scqubits.core.hilbert_space import HilbertSpace
        from scqubits.core.oscillator import Oscillator
        from scqubits.utils.spectrum_utils import identity_wrap

        resonator = Oscillator(r_f, truncated_dim=res_dim)
        fluxonium = Fluxonium(*params, flux=0.5, cutoff=cutoff, truncated_dim=qub_dim)
        hilbertspace = HilbertSpace([resonator, fluxonium])
        hilbertspace.add_interaction(
            g=g, op1=resonator.creation_operator, op2=fluxonium.n_operator, add_hc=True
        )

        def update_hilbertspace(flx: float) -> None:
            fluxonium.flux = flx

        old = scq_settings.PROGRESSBAR_DISABLED
        scq_settings.PROGRESSBAR_DISABLED = not progress
        sweep = ParameterSweep(
            hilbertspace,
            {"params": flxs},
            update_hilbertspace=update_hilbertspace,
            evals_count=res_dim * qub_dim,
            subsys_update_info={"params": [fluxonium]},
            labeling_scheme="LX",
        )
        scq_settings.PROGRESSBAR_DISABLED = old

    def get_n_oper(
        paramsweep: ParameterSweep, paramindex_tuple: Tuple[int, int], **kwargs
    ) -> NDArray[np.float64]:
        fluxonium = cast(Fluxonium, paramsweep.get_subsys(1))

        bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
        id_wrapped_op = identity_wrap(
            fluxonium.n_operator,
            fluxonium,
            paramsweep.hilbertspace.subsystem_list,
            op_in_eigenbasis=False,
            evecs=bare_evecs,
        )

        dressed_evecs = paramsweep["evecs"][paramindex_tuple]
        dressed_op_data = id_wrapped_op.transform(dressed_evecs)

        def get_idx(i: int, j: int) -> int:
            return paramsweep.dressed_index((i, j), paramindex_tuple)

        idx_0 = [get_idx(0, i) for i in range(return_dim)]
        idx_1 = [get_idx(1, j) for j in range(return_dim)]
        return dressed_op_data[np.ix_(idx_0, idx_1)]

    sweep.add_sweep(get_n_oper, "n_oper")

    return sweep, sweep["n_oper"]
