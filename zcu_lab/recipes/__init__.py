"""Explicit user recipe declarations supplied only by composition roots."""

from .amplitude_rabi import DEFINITION as AMPLITUDE_RABI
from .cpmg import DEFINITION as CPMG
from .lookback import DEFINITION as LOOKBACK
from .mist_drive_frequency import DEFINITION as MIST_DRIVE_FREQUENCY
from .mist_flux_dep import DEFINITION as MIST_FLUX_DEP
from .mist_power_trace import DEFINITION as MIST_POWER_TRACE
from .mux_amp_rabi import DEFINITION as MUX_AMP_RABI
from .mux_len_rabi import DEFINITION as MUX_LEN_RABI
from .mux_onetone import DEFINITION as MUX_ONETONE
from .mux_singleshot_ge import DEFINITION as MUX_SINGLESHOT_GE
from .mux_t1 import DEFINITION as MUX_T1
from .mux_t2echo import DEFINITION as MUX_T2ECHO
from .mux_t2ramsey import DEFINITION as MUX_T2RAMSEY
from .mux_twotone_spectrum import DEFINITION as MUX_TWOTONE_SPECTRUM
from .onetone_range_find import DEFINITION as ONETONE_RANGE_FIND
from .onetone_spectrum import DEFINITION as ONETONE_SPECTRUM
from .onetone_spectrum_over_flux import DEFINITION as ONETONE_FLUX
from .onetone_spectrum_over_power import DEFINITION as ONETONE_POWER
from .singleshot_ge import DEFINITION as SINGLESHOT_GE
from .t1 import DEFINITION as T1
from .t2echo import DEFINITION as T2ECHO
from .t2ramsey import DEFINITION as T2RAMSEY
from .time_rabi import DEFINITION as TIME_RABI
from .twotone_flux_dep import DEFINITION as TWOTONE_FLUX_DEP
from .twotone_spectrum import DEFINITION as TWOTONE_SPECTRUM

RECIPES = (
    LOOKBACK,
    MIST_POWER_TRACE,
    MIST_DRIVE_FREQUENCY,
    ONETONE_SPECTRUM,
    ONETONE_FLUX,
    ONETONE_POWER,
    SINGLESHOT_GE,
    T2RAMSEY,
    T2ECHO,
    CPMG,
    T1,
    AMPLITUDE_RABI,
    TIME_RABI,
    TWOTONE_SPECTRUM,
    MIST_FLUX_DEP,
    MUX_ONETONE,
    ONETONE_RANGE_FIND,
    TWOTONE_FLUX_DEP,
    MUX_TWOTONE_SPECTRUM,
    MUX_AMP_RABI,
    MUX_LEN_RABI,
    MUX_T1,
    MUX_T2RAMSEY,
    MUX_T2ECHO,
    MUX_SINGLESHOT_GE,
)
