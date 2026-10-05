"""Explicit user measurement declarations for the Autofluxdep composition root."""

from zcu_lab.v2.autofluxdep.lenrabi.autofluxdep import EXPERIMENT as LENRABI
from zcu_lab.v2.autofluxdep.mist.autofluxdep import EXPERIMENT as MIST
from zcu_lab.v2.autofluxdep.qubit_freq.autofluxdep import EXPERIMENT as QUBIT_FREQ
from zcu_lab.v2.autofluxdep.ro_optimize.autofluxdep import EXPERIMENT as RO_OPTIMIZE
from zcu_lab.v2.autofluxdep.t1.autofluxdep import EXPERIMENT as T1
from zcu_lab.v2.autofluxdep.t2echo.autofluxdep import EXPERIMENT as T2ECHO
from zcu_lab.v2.autofluxdep.t2ramsey.autofluxdep import EXPERIMENT as T2RAMSEY
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
