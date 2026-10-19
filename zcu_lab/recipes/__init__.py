"""Explicit user recipe declarations supplied only by composition roots."""

from zcu_tools.mcp.measure.recipe import RecipeDefinition

from .amplitude_rabi import DEFINITION as AMPLITUDE_RABI
from .lookback import DEFINITION as LOOKBACK
from .mist_flux_dep import DEFINITION as MIST_FLUX_DEP
from .mux_amp_rabi import DEFINITION as MUX_AMP_RABI
from .mux_freq import DEFINITION as MUX_FREQ
from .mux_len_rabi import DEFINITION as MUX_LEN_RABI
from .mux_singleshot_ge import DEFINITION as MUX_SINGLESHOT_GE
from .mux_t1 import DEFINITION as MUX_T1
from .mux_t2echo import DEFINITION as MUX_T2ECHO
from .mux_t2ramsey import DEFINITION as MUX_T2RAMSEY
from .onetone_spectrum import DEFINITION as ONETONE_SPECTRUM
from .onetone_spectrum_over_flux import DEFINITION as ONETONE_FLUX
from .onetone_spectrum_over_power import DEFINITION as ONETONE_POWER
from .singleshot_ge import DEFINITION as SINGLESHOT_GE
from .t1 import DEFINITION as T1
from .t2echo import DEFINITION as T2ECHO
from .t2ramsey import DEFINITION as T2RAMSEY
from .time_rabi import DEFINITION as TIME_RABI
from .twotone_spectrum import DEFINITION as TWOTONE_SPECTRUM

RECIPES: tuple[RecipeDefinition, ...] = (
    LOOKBACK,
    ONETONE_SPECTRUM,
    ONETONE_FLUX,
    ONETONE_POWER,
    SINGLESHOT_GE,
    T2RAMSEY,
    T2ECHO,
    T1,
    AMPLITUDE_RABI,
    TIME_RABI,
    TWOTONE_SPECTRUM,
    MIST_FLUX_DEP,
    MUX_FREQ,
    MUX_AMP_RABI,
    MUX_LEN_RABI,
    MUX_T1,
    MUX_T2RAMSEY,
    MUX_T2ECHO,
    MUX_SINGLESHOT_GE,
)
