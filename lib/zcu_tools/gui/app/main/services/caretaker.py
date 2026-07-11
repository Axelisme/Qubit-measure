"""Main-app snapshot codec and persistence composition."""

from __future__ import annotations

from pathlib import Path

from zcu_tools.gui.session.persistence import (
    RestoreOutcome,
    SingleFileCaretaker,
    SnapshotOriginator,
)

from .persistence_types import APP_STATE_VERSION, AppPersistedState
from .ports import RestoreReport

_STATE_FILENAME = "gui_state_v1.json"


class AppSnapshotCodec:
    def default(self) -> AppPersistedState:
        return AppPersistedState()

    def decode(self, raw: object) -> AppPersistedState:
        state = AppPersistedState.model_validate(raw)
        if state.version != APP_STATE_VERSION:
            raise ValueError(
                f"Unsupported GUI state version {state.version!r} "
                f"(expected {APP_STATE_VERSION})"
            )
        return state

    def encode(self, state: AppPersistedState) -> object:
        return state.model_dump(mode="json")


def create_persistence_caretaker(
    originator: SnapshotOriginator[AppPersistedState, RestoreReport],
    *,
    cache_dir: Path | None = None,
) -> SingleFileCaretaker[AppPersistedState, RestoreReport]:
    return SingleFileCaretaker(
        originator,
        codec=AppSnapshotCodec(),
        filename=_STATE_FILENAME,
        cache_dir=cache_dir,
    )


__all__ = [
    "AppSnapshotCodec",
    "RestoreOutcome",
    "SingleFileCaretaker",
    "create_persistence_caretaker",
]
