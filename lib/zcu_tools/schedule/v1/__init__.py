import warnings

from . import qubit, resonator
from .lookback import measure_lookback
from .qubit import (
    measure_amprabi,
    measure_fid_auto,
    measure_lenrabi,
    measure_qub_flux_dep,
    measure_qub_freq,
    measure_qub_pdr_dep,
    measure_t1,
    measure_t2echo,
    measure_t2ramsey,
)
from .resonator import (
    measure_res_flux_dep,
    measure_res_freq,
    measure_res_pdr_dep,
)

warnings.warn(
    "zcu_tools.schedule.v1 is deprecated, please use zcu_tools.schedule.v2 instead.",
    DeprecationWarning,
)

__all__ = [
    "resonator",
    "qubit",
    "measure_lookback",
    "measure_res_freq",
    "measure_res_pdr_dep",
    "measure_res_flux_dep",
    "measure_qub_freq",
    "measure_qub_pdr_dep",
    "measure_qub_flux_dep",
    "measure_lenrabi",
    "measure_amprabi",
    "measure_t1",
    "measure_t2ramsey",
    "measure_t2echo",
    "measure_fid_auto",
]
