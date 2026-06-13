"""Shared Controller base for the analysis GUI apps (fluxdep, dispersive).

A Qt-free, domain-free skeleton holding the three pieces every analysis-GUI
Controller carries: the app ``State``, an ``EventBus``, and the ``project_root``
(the base dir default result/database paths anchor under). Subclasses add their
domain services in ``__init__`` and keep their per-command façade methods
(``call a service then self._emit(SomePayload(...))``); the base only owns the
storage + the ``state`` / ``bus`` read-only properties + ``get_project_root`` +
the protected ``_emit`` helper.

The base is generic over both the State and the EventBus type, so
``self.state`` / ``self.bus`` return each app's concrete type with no casts. It
never imports a concrete ``EventBus``: the subclass resolves ``bus or
EventBus()`` for its own app and passes the concrete instance up. main-gui does
NOT inherit this (it has a different construction path and an operation
lifecycle); only the two copy-then-rewrite analysis siblings do.
"""

from __future__ import annotations

import os
from typing import Generic, TypeVar

from zcu_tools.gui.event_bus import BaseEventBus, BasePayload

StateT = TypeVar("StateT")
BusT = TypeVar("BusT", bound=BaseEventBus)


class BaseController(Generic[StateT, BusT]):
    """Storage + property skeleton shared by the analysis-GUI Controllers."""

    def __init__(
        self,
        state: StateT,
        bus: BusT,
        project_root: str | None = None,
    ) -> None:
        self._state = state
        self._bus = bus
        # Base dir the default result/database paths anchor under (the repo root,
        # injected by the entry script) so a .bat launcher that cd's into script/
        # does not scope defaults under script/. None → cwd (legacy / tests).
        self._project_root = project_root if project_root is not None else os.getcwd()

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def bus(self) -> BusT:
        return self._bus

    def get_project_root(self) -> str:
        """Base dir the default result/database paths anchor under (the repo
        root, injected by the entry script). The project dialog derives defaults
        through ``default_result_dir`` against this, NOT cwd, so a .bat launcher
        that cd's into script/ still scopes under the repo root."""
        return self._project_root

    def _emit(self, payload: BasePayload) -> None:
        """Dispatch an EventBus payload to every subscriber of its type."""
        self._bus.emit(payload)
