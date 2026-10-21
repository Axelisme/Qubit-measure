"""Single-fluxonium resource adapters; numeric owners retain all physics."""

import numpy as np

from zcu_tools.core.simulation import EnvironmentSnapshot
from zcu_tools.resources.single_fluxonium.schema import SingleFluxoniumSettings


def dc_flux(settings: SingleFluxoniumSettings, snapshot: EnvironmentSnapshot) -> float:
    """Return absolute reduced flux (Phi0) from one live native-unit DC value.

    settings defines offset, source name, A/V units and inclusive limits.
    snapshot must contain exactly that source once, with a finite value inside
    its limits. Missing, additional, duplicate or invalid values raise ValueError.
    No startup reset, unit conversion, model construction or I/O occurs.
    """
    source = settings.dc_bias.source
    if len(snapshot.values) != 1 or snapshot.values[0][0] != source.name:
        raise ValueError("dc_bias.source: requires exactly one matching round value")
    value = snapshot.values[0][1]
    if not np.isfinite(value) or not source.limits[0] <= value <= source.limits[1]:
        raise ValueError("dc_bias.source.value: nonfinite or outside source limits")
    return settings.dc_bias.offset_phi0 + source.flux_per_unit * value
