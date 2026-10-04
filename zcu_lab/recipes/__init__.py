"""Explicit user recipe declarations supplied only by composition roots."""

from zcu_tools.mcp.measure.recipe import RecipeDefinition

from .lookback import DEFINITION as LOOKBACK
from .onetone_spectrum import DEFINITION as ONETONE_SPECTRUM
from .onetone_spectrum_over_flux import DEFINITION as ONETONE_FLUX
from .onetone_spectrum_over_power import DEFINITION as ONETONE_POWER
from .singleshot_ge import DEFINITION as SINGLESHOT_GE
from .t1 import DEFINITION as T1
from .t2echo import DEFINITION as T2ECHO
from .t2ramsey import DEFINITION as T2RAMSEY

RECIPES: tuple[RecipeDefinition, ...] = (
    LOOKBACK,
    ONETONE_SPECTRUM,
    ONETONE_FLUX,
    ONETONE_POWER,
    SINGLESHOT_GE,
    T2RAMSEY,
    T2ECHO,
    T1,
)
