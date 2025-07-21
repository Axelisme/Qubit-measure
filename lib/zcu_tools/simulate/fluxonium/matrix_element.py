from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    # otherwise, lazy import
    import scqubits as scq


def calculate_n_oper(
    params: Tuple[float, float, float],
    flx: float,
    return_dim: int = 4,
    esys: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """
    Calculate the matrix elements of the fluxonium
    """

    cutoff = 30

    from scqubits import Fluxonium  # lazy import

    fluxonium = Fluxonium(*params, flux=flx, cutoff=cutoff, truncated_dim=return_dim)
    if esys is None:
        esys = fluxonium.eigensys(evals_count=return_dim)

    matrix_elements = fluxonium.n_operator(energy_esys=esys)

    return matrix_elements[:return_dim, :return_dim]


def calculate_n_oper_vs_flx(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    return_dim: int = 4,
    spectrum_data: Optional["scq.SpectrumData"] = None,
) -> Tuple["scq.SpectrumData", np.ndarray]:
    """
    Calculate the matrix elements of the fluxonium vs. a parameter
    """

    if spectrum_data is None:
        cutoff = 40

        from scqubits import Fluxonium  # lazy import

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
    flxs: np.ndarray,
    r_f: float,
    g: float,
    return_dim: int = 4,
    progress: bool = True,
    sweep: Optional["scq.ParameterSweep"] = None,
) -> Tuple["scq.ParameterSweep", np.ndarray]:
    """
    Calculate the matrix elements of the system over a parameter sweep
    """

    if sweep is None:
        cutoff = 50
        qub_dim = 20
        res_dim = 10

        import scqubits as scq  # lazy import

        resonator = scq.Oscillator(r_f, truncated_dim=res_dim)
        fluxonium = scq.Fluxonium(
            *params, flux=0.5, cutoff=cutoff, truncated_dim=qub_dim
        )
        hilbertspace = scq.HilbertSpace([resonator, fluxonium])
        hilbertspace.add_interaction(
            g=g, op1=resonator.creation_operator, op2=fluxonium.n_operator, add_hc=True
        )

        def update_hilbertspace(flx: float) -> None:
            fluxonium.flux = flx

        old = scq.settings.PROGRESSBAR_DISABLED
        scq.settings.PROGRESSBAR_DISABLED = not progress
        sweep = scq.ParameterSweep(
            hilbertspace,
            {"params": flxs},
            update_hilbertspace=update_hilbertspace,
            evals_count=res_dim * qub_dim,
            subsys_update_info={"params": [fluxonium]},
            labeling_scheme="LX",
        )
        scq.settings.PROGRESSBAR_DISABLED = old

    def get_n_oper(
        paramsweep: scq.ParameterSweep, paramindex_tuple: Tuple[int, int], **kwargs
    ) -> np.ndarray:
        fluxonium: scq.Fluxonium = paramsweep.get_subsys(1)

        bare_evecs = paramsweep["bare_evecs"]["subsys":1][paramindex_tuple]
        id_wrapped_op = scq.identity_wrap(
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
