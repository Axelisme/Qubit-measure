"""App-local service sub-packages for autofluxdep-gui.

Holds the read-only ``remote`` bridge (the RPC/MCP face onto the Controller) and
the app-local workflow persistence caretaker. The session services (connection /
context / device / startup) are the shared ones composed in the Controller, not
app-local.
"""

from __future__ import annotations

from .caretaker import PersistenceCaretaker, RestoreOutcome
from .persistence_types import (
    APP_STATE_VERSION,
    AppPersistedState,
    PersistedFluxSweep,
    PersistedNode,
    PersistedWorkflow,
    PersistenceError,
    RestoreIssue,
    RestoreReport,
)

__all__ = [
    "APP_STATE_VERSION",
    "AppPersistedState",
    "PersistedFluxSweep",
    "PersistedNode",
    "PersistedWorkflow",
    "PersistenceCaretaker",
    "PersistenceError",
    "RestoreIssue",
    "RestoreOutcome",
    "RestoreReport",
]
