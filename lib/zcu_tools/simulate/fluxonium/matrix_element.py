from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from .dressed import require_dressed_index
from .scq_settings import scq_progress

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep
    from scqubits.core.storage import SpectrumData

_SCALAR_QUB_CUTOFF = 30
_SWEEP_QUB_CUTOFF = 40
_SYSTEM_QUB_CUTOFF = 50
_SYSTEM_QUB_DIM = 20
_SYSTEM_RES_DIM = 10


def calculate_n_oper(
    params: tuple[float, float, float],
    flux: float,
    return_dim: int = 4,
    esys: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None,
) -> NDArray[np.float64]:
    """
    Calculate the matrix elements of the fluxonium
    """

    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(
        *params, flux=flux, cutoff=_SCALAR_QUB_CUTOFF, truncated_dim=return_dim
    )
    if esys is None:
        esys = fluxonium.eigensys(evals_count=return_dim)

    matrix_elements = fluxonium.n_operator(energy_esys=esys)

    return matrix_elements[:return_dim, :return_dim]


def calculate_n_oper_vs_flux(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    return_dim: int = 4,
    spectrum_data: SpectrumData | None = None,
) -> tuple[SpectrumData, NDArray[np.complex128]]:
    """
    Calculate the matrix elements of the fluxonium vs. a parameter
    """

    if spectrum_data is None or spectrum_data.matrixelem_table is None:
        from scqubits.core.fluxonium import Fluxonium

        fluxonium = Fluxonium(
            *params, flux=0.5, cutoff=_SWEEP_QUB_CUTOFF, truncated_dim=return_dim
        )
        # Pin BLAS to one thread for the sweep: on these tiny matrices OpenBLAS
        # multithreading is a net loss (thread overhead >> work), ~150x slower
        # — a benchmarked finding, same as calculate_energy_vs_flux. The context
        # manager limits only this call and restores on exit (no global side
        # effect). threadpoolctl ships with joblib (a project dependency).
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1, user_api="blas"):
            spectrum_data = fluxonium.get_matelements_vs_paramvals(
                operator="n_operator",
                param_name="flux",
                param_vals=fluxs,
                evals_count=return_dim,
            )
    matrix_elements = spectrum_data.matrixelem_table
    if matrix_elements is None:
        raise RuntimeError("scqubits did not produce n_operator matrix elements")

    return spectrum_data, matrix_elements[:, :return_dim, :return_dim]


def calculate_system_n_oper_vs_flux(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    bare_rf: float,
    g: float,
    return_dim: int = 4,
    progress: bool = True,
    sweep: ParameterSweep | None = None,
) -> tuple[ParameterSweep, NDArray[np.complex128]]:
    """
    Calculate the matrix elements of the system over a parameter sweep
    """

    if sweep is None:
        from scqubits.core.fluxonium import Fluxonium
        from scqubits.core.hilbert_space import HilbertSpace
        from scqubits.core.oscillator import Oscillator
        from scqubits.core.param_sweep import ParameterSweep

        resonator = Oscillator(bare_rf, truncated_dim=_SYSTEM_RES_DIM)
        fluxonium = Fluxonium(
            *params,
            flux=0.5,
            cutoff=_SYSTEM_QUB_CUTOFF,
            truncated_dim=_SYSTEM_QUB_DIM,
        )
        hilbertspace = HilbertSpace([resonator, fluxonium])
        hilbertspace.add_interaction(
            g=g, op1=resonator.creation_operator, op2=fluxonium.n_operator, add_hc=True
        )

        def update_hilbertspace(flux: float) -> None:
            fluxonium.flux = flux

        with scq_progress(progress):
            sweep = ParameterSweep(
                hilbertspace,
                {"params": fluxs},
                update_hilbertspace=update_hilbertspace,
                evals_count=_SYSTEM_RES_DIM * _SYSTEM_QUB_DIM,
                subsys_update_info={"params": [fluxonium]},
                labeling_scheme="LX",
            )

    def get_n_oper(
        paramsweep: ParameterSweep, paramindex_tuple: tuple[int, int], **kwargs
    ) -> NDArray[np.complex128]:
        from scqubits.utils.spectrum_utils import identity_wrap

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
            bare_state = (i, j)
            return require_dressed_index(
                paramsweep.dressed_index(bare_state, paramindex_tuple),
                bare_state,
                context=f"calculate_system_n_oper_vs_flux paramindex={paramindex_tuple}",
            )

        idx_0 = [get_idx(0, i) for i in range(return_dim)]
        idx_1 = [get_idx(1, j) for j in range(return_dim)]
        return dressed_op_data[np.ix_(idx_0, idx_1)]

    sweep.add_sweep(get_n_oper, "n_oper")

    n_oper = sweep["n_oper"]
    if n_oper is None:
        raise RuntimeError("scqubits did not produce system n_operator matrix elements")
    return sweep, np.asarray(n_oper, dtype=np.complex128)


def calculate_phi_oper(
    params: tuple[float, float, float],
    flux: float,
    return_dim: int = 4,
    esys: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None,
) -> NDArray[np.complex128]:
    """
    Calculate the matrix elements of the phi operator
    """

    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(
        *params, flux=flux, cutoff=_SCALAR_QUB_CUTOFF, truncated_dim=return_dim
    )
    if esys is None:
        esys = fluxonium.eigensys(evals_count=return_dim)

    matrix_elements = fluxonium.phi_operator(energy_esys=esys)

    return matrix_elements[:return_dim, :return_dim]


def calculate_phi_oper_vs_flux(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    return_dim: int = 4,
    spectrum_data: SpectrumData | None = None,
) -> tuple[SpectrumData, NDArray[np.complex128]]:
    """
    Calculate the matrix elements of the phi operator vs. a parameter
    """

    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(
        *params, flux=0.5, cutoff=_SWEEP_QUB_CUTOFF, truncated_dim=return_dim
    )
    phi_oper = fluxonium.phi_operator(energy_esys=False)

    if spectrum_data is None or spectrum_data.matrixelem_table is None:
        # See calculate_n_oper_vs_flux: pin BLAS to one thread for the sweep
        # (OpenBLAS multithreading is a ~150x net loss on these tiny matrices).
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1, user_api="blas"):
            spectrum_data = fluxonium.get_matelements_vs_paramvals(
                operator=phi_oper,
                param_name="flux",
                param_vals=fluxs,
                evals_count=return_dim,
            )
    matrix_elements = spectrum_data.matrixelem_table
    if matrix_elements is None:
        raise RuntimeError("scqubits did not produce phi_operator matrix elements")

    return spectrum_data, matrix_elements[:, :return_dim, :return_dim]


def calculate_sin_phi_oper(
    params: tuple[float, float, float],
    flux: float,
    return_dim: int = 4,
    esys: tuple[NDArray[np.float64], NDArray[np.complex128]] | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> NDArray[np.complex128]:
    """
    Calculate the matrix elements of the sin(phi/2) operator
    """

    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(
        *params, flux=flux, cutoff=_SCALAR_QUB_CUTOFF, truncated_dim=return_dim
    )
    if esys is None:
        esys = fluxonium.eigensys(evals_count=return_dim)

    matrix_elements = fluxonium.sin_phi_operator(
        alpha=alpha, beta=beta, energy_esys=esys
    )

    return matrix_elements[:return_dim, :return_dim]


def calculate_sin_phi_oper_vs_flux(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    return_dim: int = 4,
    spectrum_data: SpectrumData | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> tuple[SpectrumData, NDArray[np.complex128]]:
    """
    Calculate the matrix elements of the sin(phi/2) operator vs. a parameter
    """

    from scqubits.core.fluxonium import Fluxonium

    fluxonium = Fluxonium(
        *params, flux=0.5, cutoff=_SWEEP_QUB_CUTOFF, truncated_dim=return_dim
    )
    sin_phi_oper = fluxonium.sin_phi_operator(alpha=alpha, beta=beta, energy_esys=False)

    if spectrum_data is None or spectrum_data.matrixelem_table is None:
        # See calculate_n_oper_vs_flux: pin BLAS to one thread for the sweep
        # (OpenBLAS multithreading is a ~150x net loss on these tiny matrices).
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1, user_api="blas"):
            spectrum_data = fluxonium.get_matelements_vs_paramvals(
                operator=sin_phi_oper,
                param_name="flux",
                param_vals=fluxs,
                evals_count=return_dim,
            )
    matrix_elements = spectrum_data.matrixelem_table
    if matrix_elements is None:
        raise RuntimeError("scqubits did not produce sin_phi_operator matrix elements")

    return spectrum_data, matrix_elements[:, :return_dim, :return_dim]
