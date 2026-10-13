"""Public command parameter declarations for sampling and Filter consumers.

SET_THRESHOLD_PARAMS describes OneTone's required dimensionless prominence.
SET_SETTINGS_PARAMS describes TwoTone's optional dimensionless detector settings;
at least one non-null value is required, and sigma limits depend on smooth_method
as stated in the sigma declaration. STROKE_PARAMS describes nonempty native x/GHz
vertices, a normalized radius and explicit select/erase mode. x is device position
for sampling and calibrated flux for Filter. SET_MIN_DISTANCE_PARAMS describes
Filter's required normalized distance. ParamSpec carries inclusive numeric limits
and dependent-limit descriptions from the domain owner; schemas do not validate
those domains. Consumers use schema_property without constructing a live plugin.
"""

from zcu_tools.gui.app.fluxdep.brush_commands import STROKE_PARAMS as STROKE_PARAMS
from zcu_tools.gui.app.fluxdep.cross_selection import (
    SET_MIN_DISTANCE_PARAMS as SET_MIN_DISTANCE_PARAMS,
)
from zcu_tools.gui.app.fluxdep.onetone import (
    SET_THRESHOLD_PARAMS as SET_THRESHOLD_PARAMS,
)
from zcu_tools.gui.app.fluxdep.twotone import SET_SETTINGS_PARAMS as SET_SETTINGS_PARAMS

__all__ = [
    "SET_THRESHOLD_PARAMS",
    "SET_SETTINGS_PARAMS",
    "STROKE_PARAMS",
    "SET_MIN_DISTANCE_PARAMS",
]
