"""Explicit user measurement declarations for the Autofluxdep composition root."""

from zcu_tools.experiment.v2_gui.autofluxdep.lenrabi import EXPERIMENT as LENRABI
from zcu_tools.experiment.v2_gui.autofluxdep.mist import EXPERIMENT as MIST
from zcu_tools.experiment.v2_gui.autofluxdep.qubit_freq import EXPERIMENT as QUBIT_FREQ
from zcu_tools.experiment.v2_gui.autofluxdep.ro_optimize import (
    EXPERIMENT as RO_OPTIMIZE,
)
from zcu_tools.experiment.v2_gui.autofluxdep.t1 import EXPERIMENT as T1
from zcu_tools.experiment.v2_gui.autofluxdep.t2echo import EXPERIMENT as T2ECHO
from zcu_tools.experiment.v2_gui.autofluxdep.t2ramsey import EXPERIMENT as T2RAMSEY
from zcu_tools.gui.app.autofluxdep.catalog import ExperimentCatalog


def build_catalog() -> ExperimentCatalog:
    """Build the ordered catalog of this package's current measurement Builders.

    Return a fresh immutable catalog over the stateless declaration singletons.
    Its order controls the add menu, not saved workflow execution order.
    Propagate declaration validation errors; do not register at import time.
    """
    return ExperimentCatalog(
        (QUBIT_FREQ, LENRABI, RO_OPTIMIZE, T1, T2RAMSEY, T2ECHO, MIST)
    )
