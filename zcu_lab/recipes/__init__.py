"""Explicit user recipe declarations supplied only by composition roots."""

from .amplitude_rabi import DEFINITION as AMPLITUDE_RABI
from .chain_amp_rabi import DEFINITION as CHAIN_AMP_RABI
from .chain_len_rabi import DEFINITION as CHAIN_LEN_RABI
from .chain_t1 import DEFINITION as CHAIN_T1
from .chain_t2echo import DEFINITION as CHAIN_T2ECHO
from .chain_t2ramsey import DEFINITION as CHAIN_T2RAMSEY
from .cpmg import DEFINITION as CPMG
from .lookback import DEFINITION as LOOKBACK
from .mist_drive_frequency import DEFINITION as MIST_DRIVE_FREQUENCY
from .mist_flux_dep import DEFINITION as MIST_FLUX_DEP
from .mist_power_trace import DEFINITION as MIST_POWER_TRACE
from .mux_amp_rabi import DEFINITION as MUX_AMP_RABI
from .mux_len_rabi import DEFINITION as MUX_LEN_RABI
from .mux_onetone import DEFINITION as MUX_ONETONE
from .mux_onetone_flux_global import DEFINITION as MUX_ONETONE_FLUX_GLOBAL
from .mux_onetone_flux_local import DEFINITION as MUX_ONETONE_FLUX_LOCAL
from .mux_singleshot_ge import DEFINITION as MUX_SINGLESHOT_GE
from .mux_t1 import DEFINITION as MUX_T1
from .mux_t2echo import DEFINITION as MUX_T2ECHO
from .mux_t2ramsey import DEFINITION as MUX_T2RAMSEY
from .mux_twotone_flux_global import DEFINITION as MUX_TWOTONE_FLUX_GLOBAL
from .mux_twotone_flux_local import DEFINITION as MUX_TWOTONE_FLUX_LOCAL
from .mux_twotone_spectrum import DEFINITION as MUX_TWOTONE_SPECTRUM
from .onetone_range_find import DEFINITION as ONETONE_RANGE_FIND
from .onetone_spectrum import DEFINITION as ONETONE_SPECTRUM
from .onetone_spectrum_over_flux import DEFINITION as ONETONE_FLUX
from .onetone_spectrum_over_power import DEFINITION as ONETONE_POWER
from .raman_gf import DEFINITION as RAMAN_GF
from .ro_optimize_chain_freq import DEFINITION as RO_OPTIMIZE_CHAIN_FREQ
from .ro_optimize_chain_length import DEFINITION as RO_OPTIMIZE_CHAIN_LENGTH
from .ro_optimize_chain_power import DEFINITION as RO_OPTIMIZE_CHAIN_POWER
from .singleshot_ge import DEFINITION as SINGLESHOT_GE
from .stirap_amplitude import DEFINITION as STIRAP_AMPLITUDE
from .stirap_delay import DEFINITION as STIRAP_DELAY
from .stirap_detuning import DEFINITION as STIRAP_DETUNING
from .t1 import DEFINITION as T1
from .t1_cov import DEFINITION as T1_COV
from .t2echo import DEFINITION as T2ECHO
from .t2ramsey import DEFINITION as T2RAMSEY
from .time_rabi import DEFINITION as TIME_RABI
from .two_qubit_constructive_phase import DEFINITION as TWO_QUBIT_CONSTRUCTIVE_PHASE
from .two_qubit_cz_amplitude import DEFINITION as TWO_QUBIT_CZ_AMPLITUDE
from .two_qubit_local_z import DEFINITION as TWO_QUBIT_LOCAL_Z
from .twotone_chain import DEFINITION as TWOTONE_CHAIN
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
    T1_COV,
    AMPLITUDE_RABI,
    TIME_RABI,
    TWOTONE_SPECTRUM,
    TWOTONE_CHAIN,
    CHAIN_AMP_RABI,
    CHAIN_LEN_RABI,
    CHAIN_T1,
    CHAIN_T2RAMSEY,
    CHAIN_T2ECHO,
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
    MUX_ONETONE_FLUX_GLOBAL,
    MUX_ONETONE_FLUX_LOCAL,
    MUX_TWOTONE_FLUX_GLOBAL,
    MUX_TWOTONE_FLUX_LOCAL,
    STIRAP_AMPLITUDE,
    STIRAP_DELAY,
    STIRAP_DETUNING,
    RAMAN_GF,
    RO_OPTIMIZE_CHAIN_POWER,
    RO_OPTIMIZE_CHAIN_FREQ,
    RO_OPTIMIZE_CHAIN_LENGTH,
    TWO_QUBIT_CZ_AMPLITUDE,
    TWO_QUBIT_LOCAL_Z,
    TWO_QUBIT_CONSTRUCTIVE_PHASE,
)
