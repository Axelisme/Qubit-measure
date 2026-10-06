"""Shared optimistic-concurrency version table for the GUI apps.

App-agnostic, import-clean (stdlib only): a monotonic per-resource version
counter every GUI app (``app/measure`` / ``app/fluxdep`` / ``app/dispersive`` /
``app/autofluxdep`` via ``SessionState``) uses to guard against concurrent
edits. The resource KEYS are domain-specific (each app names its own
``context`` / ``tab:<id>`` / ``spectrum:<name>`` / ... keys next to its own
``*_VERSION_KEY`` constants), but the counter mechanism is identical, so it lives
here once.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class VersionTable:
    """Monotonic per-resource version counters (optimistic-concurrency guard).

    Resource owners call ``bump`` on their owner thread after semantic writes.
    Guards compare observed versions with this passive container in the same
    owner turn. Keys are app-specific strings; absent resources read as 0.

    ``retire`` removes one live key but preserves its counter for same-key
    recreation. ``drop_prefix`` forgets matching live and retired counters,
    allowing them to restart at 1; use it only when key reuse cannot make an
    old observation match a new resource. Each app documents its key lifecycle
    beside its ``*_VERSION_KEY`` constants in ``state.py``.
    """

    def __init__(self) -> None:
        self._versions: dict[str, int] = {}
        self._retired_versions: dict[str, int] = {}

    def bump(self, key: str) -> int:
        """Advance a resource's version (a semantic write happened).

        Return 1 for a never-bumped or explicitly forgotten key. A live or
        retired key advances from its last version. Owners must retire or
        forget resources at teardown so absent resources read as 0.
        """
        previous = self._versions.get(key, 0)
        if key in self._retired_versions:
            previous = self._retired_versions.pop(key)
        new = previous + 1
        self._versions[key] = new
        logger.debug("version bump: %s -> %d", key, new)
        return new

    def get(self, key: str) -> int:
        """Current live version of ``key`` (0 if never bumped or removed)."""
        return self._versions.get(key, 0)

    def snapshot(self) -> dict[str, int]:
        """Detached live-key version copy; retired keys are omitted."""
        return dict(self._versions)

    def retire(self, key: str) -> None:
        """Retire one complete literal key, retaining its last counter.

        ``get(key)`` then returns 0 and ``snapshot()`` omits it. The next
        ``bump(key)`` continues above its last live version, so a re-created
        resource cannot match an observation from before retirement. Repeated
        retirement and unknown keys are no-ops. Retained counters live only as
        long as this table; ``drop_prefix`` explicitly forgets them.
        """
        if key in self._versions:
            self._retired_versions[key] = self._versions.pop(key)
            logger.debug("version retire: %s", key)

    def drop_prefix(self, prefix: str) -> None:
        """Forget every key starting with ``prefix`` (e.g. a closed tab).

        Matching live keys read as 0 and disappear from ``snapshot``. Retired
        counters also disappear; the next ``bump`` restarts at 1. This is a
        forgetting operation, unlike ``retire``. ``prefix`` is literal, and an
        empty prefix forgets all keys.
        """
        keys = self._versions.keys() | self._retired_versions.keys()
        doomed = [k for k in keys if k.startswith(prefix)]
        for k in doomed:
            self._versions.pop(k, None)
            self._retired_versions.pop(k, None)
        if doomed:
            logger.debug("version drop_prefix: %s -> dropped %s", prefix, doomed)
