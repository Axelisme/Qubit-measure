"""Main-app snapshot codec and persistence composition."""

from __future__ import annotations

from pathlib import Path

from zcu_tools.gui.session.persistence import (
    RestoreOutcome,
    SingleFileCaretaker,
    SnapshotOriginator,
)

from .persistence_types import APP_STATE_VERSION, AppPersistedState

_STATE_FILENAME = "measure_layout_v1.json"


class AppSnapshotCodec:
    """Pure Qt layout codec; validation/version failures raise ValueError.

    Values are detached and require no Qt owner thread. No session domain state
    or legacy file migration belongs to this codec.
    """

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
    originator: SnapshotOriginator[AppPersistedState, None],
    *,
    cache_dir: Path | None = None,
) -> SingleFileCaretaker[AppPersistedState, None]:
    """Build the layout-only caretaker for an owner-thread originator.

    cache_dir is an optional caller-owned cache root; None uses the user cache.
    The returned caretaker owns atomic file I/O, not the originator or its view.
    Its restore returns no domain report. File/codec failures follow the shared
    SingleFileCaretaker error contract. No file is read during construction.
    """
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
