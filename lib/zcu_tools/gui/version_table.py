"""Shared optimistic-concurrency version table for the GUI apps.

App-agnostic, import-clean (stdlib only): a monotonic per-resource version
counter every GUI app (``app/main`` / ``app/fluxdep`` / ``app/dispersive`` /
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

    A passive container: each resource key maps to an integer that only ever
    increases by one per mutation. Callers (the resource-owning service, on the
    Qt main thread) ``bump`` a key when they actually write that resource's
    state. The guard compares an op's declared ``expected_versions`` against the
    current table atomically inside the main-thread dispatch sequence.

    Resource keys are app-specific and mid-grained; a key absent from the table
    means version 0 (never bumped, or its resource was dropped — both read as
    "gone" by the guard). Each app documents its own key set and its bump↔drop
    pairing contract beside its ``*_VERSION_KEY`` constants in its ``state.py``.
    """

    def __init__(self) -> None:
        self._versions: dict[str, int] = {}

    def bump(self, key: str) -> int:
        """Advance a resource's version (a semantic write happened).

        PAIRING CONTRACT — every resource that gets bumped must be dropped by its
        owner's teardown, or a stale dependency would spuriously match a retained
        version. Each app's state.py documents its concrete bump↔drop map; adding
        a new bumped key means adding its drop to the owner's teardown.
        """
        new = self._versions.get(key, 0) + 1
        self._versions[key] = new
        logger.debug("version bump: %s -> %d", key, new)
        return new

    def get(self, key: str) -> int:
        """Current version of ``key`` (0 if never bumped / dropped)."""
        return self._versions.get(key, 0)

    def snapshot(self) -> dict[str, int]:
        """Full table copy (the ``resources.versions`` RPC payload)."""
        return dict(self._versions)

    def drop_prefix(self, prefix: str) -> None:
        """Forget every key starting with ``prefix`` (e.g. a closed tab).

        A dependency on a dropped key reads as version 0, which the guard
        treats as stale (the resource the caller depended on is gone).
        """
        doomed = [k for k in self._versions if k.startswith(prefix)]
        for k in doomed:
            del self._versions[k]
        if doomed:
            logger.debug("version drop_prefix: %s -> dropped %s", prefix, doomed)
