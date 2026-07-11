"""Qt-free guard for state aggregates owned by one execution thread."""

from __future__ import annotations

import threading


class OwnerThreadGuard:
    """Capture an aggregate's owner thread and assert every semantic write."""

    __slots__ = ("_owner_thread_id",)

    def __init__(self) -> None:
        self._owner_thread_id = threading.get_ident()

    @property
    def owner_thread_id(self) -> int:
        return self._owner_thread_id

    def assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("State mutation must run on its owner thread")


__all__ = ["OwnerThreadGuard"]
