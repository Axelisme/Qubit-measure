"""TaskboardStore — the file-backed claim store over ``task_plans/taskboard.json``.

NOTE: This module uses a cross-process advisory file lock for atomic
read-modify-write.  The lock backend is selected per platform (``fcntl.flock`` on
POSIX, ``msvcrt.locking`` on Windows) by ``_acquire_lock`` / ``_release_lock``, so
the store works on Linux, macOS, and Windows.

Design:
  - Pure functions (no file I/O) operate on a plain ``dict`` state and are the
    primary test target.  File I/O is isolated in ``_with_lock`` which wraps each
    mutating call in an exclusive lock, writes back JSON, and re-renders the
    markdown view.
  - Read-only calls (``check``, ``list``) acquire a shared lock on POSIX so they
    never block each other but block writers.  Windows has no shared-lock mode, so
    they take the same exclusive lock there — readers serialise, but the
    critical-section guarantee is identical.
  - ``claim_id`` is an 8-character hex prefix of a UUID4 — short and collision-safe
    across any realistic number of concurrent agents.
  - ``paths`` accepts repo-relative file/directory paths, glob patterns
    (``*``/``**`` via ``fnmatch``), and ``@``-prefixed resource tokens
    (``@hw/zcu216``, ``@port/8767``).  Resource tokens use the same
    segment-wise prefix/overlap rules as ordinary paths.
  - The ``released`` history retains the last ``_HISTORY_KEEP`` entries.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

# Keep at most this many released claims in the history section.
_HISTORY_KEEP = 20
# Default TTL (seconds) for a claim with no heartbeat.
_DEFAULT_TTL: float = 2 * 3600.0  # 2 hours
# Max timeout_s accepted by taskboard_wait (server enforces this cap).
MAX_WAIT_TIMEOUT: float = 30.0

# ---------------------------------------------------------------------------
# Type alias for the mutable state dict persisted to JSON
# ---------------------------------------------------------------------------

State = dict[str, Any]
Claim = dict[str, Any]


# ---------------------------------------------------------------------------
# Path normalisation + overlap
# ---------------------------------------------------------------------------


def normalize_path(p: str) -> str:
    """Normalise a path to a canonical form for overlap comparison.

    - ``@``-prefixed resource tokens are lowercased and stripped; ``..`` and
      redundant separators are left as-is (token semantics, not filesystem paths).
    - Ordinary paths are normalised as repo-relative POSIX strings using
      ``os.path.normpath``, which folds ``.``, ``..``, and duplicate slashes
      without touching the filesystem.  The leading ``/`` is then stripped so the
      result is always repo-relative.  Dotfile components (e.g. ``.github``,
      ``.claude``) are preserved intact — only the charset-stripping approach used
      by ``str.lstrip`` is avoided.

    Raises ``ValueError`` for empty strings.
    """
    p = p.strip()
    if not p:
        raise ValueError("empty path")
    if p.startswith("@"):
        # Resource token: lowercase, strip trailing slash only (no normpath —
        # token segments are semantic, not filesystem paths).
        token = p.lower().rstrip("/")
        # Must have at least one segment after '@'.
        if token == "@" or token.endswith("@"):
            raise ValueError(f"invalid resource token: {p!r}")
        return token
    # Ordinary path: normpath handles '.', '..', duplicate slashes correctly.
    # Convert to forward slashes (POSIX) after normpath (which uses os.sep).
    normed = os.path.normpath(p).replace(os.sep, "/")
    # Strip leading slash to make it repo-relative; a bare '.' means root.
    normed = normed.lstrip("/")
    return normed or "."


def _segments(p: str) -> list[str]:
    """Split a normalised path into segments for prefix comparison."""
    return p.split("/")


# Characters that unambiguously signal a glob pattern.
# We deliberately exclude '[' because bare square brackets appear in real
# filenames (e.g. 'file[1].py') and would produce false-positive matches.
# '[' is only meaningful as a glob meta when combined with '*' or '?', which
# are themselves sufficient signals.
_GLOB_WILDCARD_CHARS = frozenset("*?")


def _is_glob(p: str) -> bool:
    """Return True when *p* is an unambiguous glob pattern (contains ``*`` or ``?``).

    Bare ``[`` in a path is treated as a literal character — it is common in
    real filenames and must not trigger fnmatch to avoid false-positive overlaps.
    """
    return bool(_GLOB_WILDCARD_CHARS.intersection(p))


def paths_overlap(a: str, b: str) -> bool:
    """Return True when normalised paths ``a`` and ``b`` overlap.

    Overlap is defined as:
      - exact match,
      - one path is an ancestor directory of the other (segment-wise prefix),
      - glob intersection when at least one path is a glob pattern
        (supports ``*`` and ``**`` via ``fnmatch``).

    Literal paths (no glob metacharacters) are **never** passed through
    ``fnmatch``, so characters like ``[`` and ``?`` in real filenames cannot
    cause false-positive matches.

    Resource tokens (``@``) always use the literal segment comparison path —
    token semantics must not be affected by glob metacharacter scanning.

    Bias toward false-positive (conservative) rather than false-negative, per
    the "rather block more than miss a conflict" invariant.
    """
    if a == b:
        return True

    # Ancestor-directory check (segment-wise, not string prefix).
    segs_a = _segments(a)
    segs_b = _segments(b)
    shorter, longer = (
        (segs_a, segs_b) if len(segs_a) <= len(segs_b) else (segs_b, segs_a)
    )
    if longer[: len(shorter)] == shorter:
        return True

    # Tokens (@-prefixed) never use fnmatch — literal segment match only.
    if a.startswith("@") or b.startswith("@"):
        return False

    # Glob intersection: only invoke fnmatch when at least one path is a pattern.
    if _is_glob(a) or _is_glob(b):
        if fnmatch.fnmatch(b, a) or fnmatch.fnmatch(a, b):
            return True

    return False


def claims_conflict(c1: Claim, c2: Claim) -> bool:
    """Return True when two claims conflict.

    Conflict requires: (a) the two claims have *different* identities, (b) at
    least one path pair from each claim overlaps, AND (c) at least one of the two
    claims has mode ``write`` (read+read is non-conflicting).

    The identity gate (a) is what lets an orchestrator and its sub-agents — all
    sharing one ``CLAUDE_CODE_SESSION_ID`` — never block each other: same identity
    means "the same caller", and a caller cannot conflict with itself (ADR-0022).
    Identity is stored under the ``identity`` key; when absent (legacy / not yet
    set) it falls back to ``owner`` so the gate degrades to per-owner coordination.
    """
    if _identity_of(c1) == _identity_of(c2):
        return False
    if c1["mode"] == "read" and c2["mode"] == "read":
        return False
    for p1 in c1["paths"]:
        for p2 in c2["paths"]:
            if paths_overlap(p1, p2):
                return True
    return False


# Sentinel identity for a ``check`` with neither a server session id nor an owner:
# it never equals any stored identity, so every overlapping grant is reported.
_NO_IDENTITY_SENTINEL = "\x00<no-identity>"


def _identity_of(claim: Claim) -> str:
    """Return a claim's conflict identity — the ``identity`` field if present,
    else ``owner`` (back-compat / fallback when no CC session id was available).
    """
    return claim.get("identity") or claim["owner"]


def _claim_covers(claim: Claim, paths: list[str], mode: str) -> bool:
    """Return True when an existing *granted* ``claim`` already covers (paths, mode).

    Coverage means every requested path is contained in (overlaps as a subset of)
    some path the claim holds, and the claim's mode is strong enough — a held
    ``write`` covers a requested ``read`` or ``write``; a held ``read`` covers only
    a requested ``read``.  Used for re-claim idempotency: a same-identity re-claim
    whose scope is already held returns the existing claim instead of adding a new
    one (ADR-0022).
    """
    if mode == "write" and claim["mode"] == "read":
        return False
    return all(any(_path_subset(req, held) for held in claim["paths"]) for req in paths)


def _path_subset(child: str, parent: str) -> bool:
    """Return True when normalised ``child`` is covered by ``parent``.

    Coverage = exact match, ``parent`` is an ancestor directory of ``child``, or
    ``parent`` is a glob that matches ``child``.  This is the directional ("is the
    requested path inside the held path?") form of :func:`paths_overlap`, which is
    symmetric.  A held ``lib/foo`` covers a re-claim of ``lib/foo/bar.py`` but a
    held ``lib/foo/bar.py`` does **not** cover a re-claim of the whole ``lib/foo``.
    """
    if child == parent:
        return True
    segs_child = _segments(child)
    segs_parent = _segments(parent)
    if (
        len(segs_parent) <= len(segs_child)
        and segs_child[: len(segs_parent)] == segs_parent
    ):
        return True
    # Tokens never use fnmatch — literal segment match only.
    if child.startswith("@") or parent.startswith("@"):
        return False
    if _is_glob(parent):
        return fnmatch.fnmatch(child, parent)
    return False


# ---------------------------------------------------------------------------
# Mode validation
# ---------------------------------------------------------------------------

_VALID_MODES: frozenset[str] = frozenset({"read", "write"})


def _validate_mode(mode: str) -> None:
    """Raise ValueError immediately if *mode* is not a recognised lock mode.

    Centralised so that all public entry points (claim, check) share the same
    fast-fail behaviour and error message.
    """
    if mode not in _VALID_MODES:
        valid = ", ".join(sorted(_VALID_MODES))
        raise ValueError(f"mode must be one of {{{valid}}}, got {mode!r}")


# ---------------------------------------------------------------------------
# Pure state-transition functions (no I/O)
# ---------------------------------------------------------------------------


def _new_claim_id() -> str:
    return uuid4().hex[:8]


def _now() -> float:
    return time.time()


def compute_conflicts(
    state: State,
    paths: list[str],
    mode: str,
    identity: str,
    exclude_id: str | None = None,
) -> list[Claim]:
    """Return all currently *granted* claims that conflict with the given claim.

    ``identity`` is the conflict identity of the prospective claim; granted claims
    sharing that identity never conflict (handled inside :func:`claims_conflict`),
    so a caller is never blocked by itself or its own sub-agents (ADR-0022).
    ``exclude_id`` skips one claim (used when re-evaluating a pending claim's own
    blockers after a release).
    """
    probe: Claim = {
        "paths": paths,
        "mode": mode,
        "status": "granted",
        "claim_id": "",
        "identity": identity,
    }
    return [
        c
        for c in state.get("claims", [])
        if c["status"] == "granted"
        and c["claim_id"] != exclude_id
        and claims_conflict(probe, c)
    ]


def add_claim(
    state: State,
    owner: str,
    paths: list[str],
    task: str,
    mode: str,
    identity: str | None = None,
    now: float | None = None,
) -> tuple[State, Claim]:
    """Attempt to add a new claim; grant immediately or queue as pending.

    ``identity`` is the conflict identity (CC-session-derived; see ADR-0022); when
    omitted it falls back to ``owner`` so callers that do not thread an identity
    coordinate per-owner.  ``owner`` itself stays a plain human-readable label.

    Re-claim idempotency: when the requested (paths, mode) is already fully covered
    by an existing *granted* claim of the same identity, that claim is returned
    unchanged (same ``claim_id``, still granted) and no new claim is added.  This
    makes a re-claim of an already-held scope a no-op rather than a duplicate.

    Returns ``(new_state, claim_record)``.  The caller decides whether to persist.
    Fast-fails on invalid mode or empty paths (caller must normalise paths first).
    """
    if mode not in ("read", "write"):
        raise ValueError(f"mode must be 'read' or 'write', got {mode!r}")
    if not paths:
        raise ValueError("paths must be non-empty")

    eff_identity = identity if identity is not None else owner

    existing = next(
        (
            c
            for c in state.get("claims", [])
            if c["status"] == "granted"
            and _identity_of(c) == eff_identity
            and _claim_covers(c, paths, mode)
        ),
        None,
    )
    if existing is not None:
        # Idempotent re-claim: scope already held by this identity — return as-is.
        return state, existing

    ts = now if now is not None else _now()
    claim_id = _new_claim_id()
    conflicts = compute_conflicts(state, paths, mode, eff_identity)
    status = "pending" if conflicts else "granted"
    blockers = [c["claim_id"] for c in conflicts]

    claim: Claim = {
        "claim_id": claim_id,
        "owner": owner,
        "identity": eff_identity,
        "paths": paths,
        "mode": mode,
        "task": task,
        "status": status,
        "blockers": blockers,
        "created": ts,
        "touched": ts,
    }
    new_claims = list(state.get("claims", []))
    new_claims.append(claim)
    released = list(state.get("released", []))
    return {**state, "claims": new_claims, "released": released}, claim


def _try_promote(state: State) -> State:
    """FIFO promote: scan pending claims in insertion order; grant those whose
    blockers are all gone (not granted by any remaining claim).  Repeat until
    no more promotions are possible (chain promotions for multi-pending queues).
    """
    changed = True
    claims = list(state["claims"])
    while changed:
        changed = False
        for i, c in enumerate(claims):
            if c["status"] != "pending":
                continue
            # Re-check conflicts against the current set of granted claims using
            # this pending claim's own identity, so promotion stays consistent with
            # the identity-aware grant decision made at claim time (ADR-0022).
            granted_set = [g for g in claims if g["status"] == "granted"]
            probe: Claim = {
                "paths": c["paths"],
                "mode": c["mode"],
                "status": "granted",
                "claim_id": "",
                "identity": _identity_of(c),
            }
            still_blocked = [g for g in granted_set if claims_conflict(probe, g)]
            if not still_blocked:
                claims[i] = {
                    **c,
                    "status": "granted",
                    "blockers": [],
                    "touched": _now(),
                }
                changed = True
    return {**state, "claims": claims}


def release_claim(state: State, claim_id: str) -> tuple[State, list[Claim]]:
    """Mark a claim as released and FIFO-promote newly unblocked pending claims.

    Returns ``(new_state, promoted_claims)`` where ``promoted_claims`` is the list
    of claims that were newly granted.
    Fast-fails if ``claim_id`` is unknown.
    """
    claims = list(state.get("claims", []))
    idx = next((i for i, c in enumerate(claims) if c["claim_id"] == claim_id), None)
    if idx is None:
        raise ValueError(f"unknown claim_id: {claim_id!r}")

    released_claim = {**claims[idx], "status": "released", "touched": _now()}
    claims.pop(idx)

    # Move to history, capped at _HISTORY_KEEP.
    released_history = list(state.get("released", []))
    released_history.append(released_claim)
    released_history = released_history[-_HISTORY_KEEP:]

    before_ids = {c["claim_id"] for c in claims if c["status"] == "granted"}
    new_state = _try_promote({**state, "claims": claims, "released": released_history})
    after_ids = {c["claim_id"] for c in new_state["claims"] if c["status"] == "granted"}
    newly_granted_ids = after_ids - before_ids
    promoted = [c for c in new_state["claims"] if c["claim_id"] in newly_granted_ids]

    return new_state, promoted


def touch_claim(state: State, claim_id: str) -> tuple[State, Claim]:
    """Update the ``touched`` timestamp of a claim to extend its TTL.

    Fast-fails if ``claim_id`` is unknown or the claim is already released.
    """
    claims = list(state.get("claims", []))
    idx = next((i for i, c in enumerate(claims) if c["claim_id"] == claim_id), None)
    if idx is None:
        raise ValueError(f"unknown claim_id: {claim_id!r}")
    if claims[idx]["status"] == "released":
        raise ValueError(f"claim {claim_id!r} is already released")
    claims[idx] = {**claims[idx], "touched": _now()}
    return {**state, "claims": claims}, claims[idx]


def reclaim_stale(
    state: State,
    now: float | None = None,
    ttl: float = _DEFAULT_TTL,
) -> tuple[State, list[Claim]]:
    """Mark stale granted/pending claims as released and cascade-promote.

    A claim is stale when ``now - touched >= ttl``.  Returns
    ``(new_state, stale_claims)`` — the reclaimed entries (already released in
    new_state).  Called lazily at the start of each mutating operation so the
    store self-cleans without a background thread.
    """
    ts = now if now is not None else _now()
    stale_ids: list[str] = []
    for c in state.get("claims", []):
        if c["status"] in ("granted", "pending") and (ts - c["touched"]) >= ttl:
            stale_ids.append(c["claim_id"])

    new_state = state
    stale: list[Claim] = []
    for cid in stale_ids:
        new_state, _ = release_claim(new_state, cid)
        # The released claim is now in new_state["released"]; retrieve it.
        entry = next((c for c in new_state["released"] if c["claim_id"] == cid), None)
        if entry is not None:
            stale.append(entry)

    return new_state, stale


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def render_markdown(state: State) -> str:
    """Produce a human-readable markdown view of the current taskboard state.

    This output is written to ``task_plans/taskboard.md`` after each mutating
    operation and is read-only for humans — the JSON store is the source of truth.
    """
    lines: list[str] = []
    ts_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("# Taskboard — 平行 agent 認領看板")
    lines.append("")
    lines.append(
        "> Auto-rendered by the taskboard MCP server from `task_plans/taskboard.json`. "
        "Do not edit manually — changes will be overwritten."
        f"  Last updated: {ts_str}."
    )
    lines.append("")

    claims: list[Claim] = state.get("claims", [])
    granted = [c for c in claims if c["status"] == "granted"]
    pending = [c for c in claims if c["status"] == "pending"]
    released = list(state.get("released", []))[-_HISTORY_KEEP:]

    # --- Active claims (granted) ---
    lines.append("## Active claims (granted)")
    lines.append("")
    if granted:
        lines.append("| claim_id | owner | mode | paths | task | since |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for c in granted:
            paths_str = "<br>".join(c["paths"])
            since = datetime.fromtimestamp(c["created"], tz=timezone.utc).strftime(
                "%m-%d %H:%M"
            )
            lines.append(
                f"| `{c['claim_id']}` | {c['owner']} | {c['mode']} "
                f"| {paths_str} | {c['task']} | {since} |"
            )
    else:
        lines.append("_No active claims._")
    lines.append("")

    # --- Pending queue ---
    lines.append("## Pending queue")
    lines.append("")
    if pending:
        lines.append("| # | claim_id | owner | mode | paths | task | blocking |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for i, c in enumerate(pending, start=1):
            paths_str = "<br>".join(c["paths"])
            blockers_str = ", ".join(f"`{b}`" for b in c["blockers"]) or "—"
            lines.append(
                f"| {i} | `{c['claim_id']}` | {c['owner']} | {c['mode']} "
                f"| {paths_str} | {c['task']} | {blockers_str} |"
            )
    else:
        lines.append("_No pending claims._")
    lines.append("")

    # --- Recent released ---
    lines.append("## Recent released")
    lines.append("")
    if released:
        recent = list(reversed(released))  # newest first
        lines.append("| claim_id | owner | mode | task | released |")
        lines.append("| --- | --- | --- | --- | --- |")
        for c in recent:
            when = datetime.fromtimestamp(c["touched"], tz=timezone.utc).strftime(
                "%m-%d %H:%M"
            )
            lines.append(
                f"| `{c['claim_id']}` | {c['owner']} | {c['mode']} "
                f"| {c['task']} | {when} |"
            )
    else:
        lines.append("_No recent history._")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Empty initial state factory
# ---------------------------------------------------------------------------


def empty_state() -> State:
    return {"claims": [], "released": []}


# ---------------------------------------------------------------------------
# Cross-platform advisory file lock
# ---------------------------------------------------------------------------
#
# ``fcntl`` is POSIX-only and ``msvcrt`` is Windows-only; importing the wrong one
# raises ``ModuleNotFoundError``.  Guard the import on ``sys.platform`` so this
# module imports cleanly on every OS, and expose a single (acquire/release) pair
# with uniform "exclusive lock around the critical section" semantics.

if sys.platform == "win32":
    import msvcrt

    # Lock a single byte at offset 0; locking past EOF is allowed on Windows, so
    # the empty lock file needs no priming.  ``msvcrt`` has no shared-lock mode —
    # both readers and writers take this exclusive lock.
    def _acquire_lock(lock_file: BinaryIO, exclusive: bool) -> None:
        lock_file.seek(0)
        # LK_LOCK blocks ~10 s (10 internal retries) then raises; loop so the wait
        # is truly blocking, matching fcntl.flock's blocking acquire.
        while True:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                time.sleep(0.1)

    def _release_lock(lock_file: BinaryIO) -> None:
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _acquire_lock(lock_file: BinaryIO, exclusive: bool) -> None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def _release_lock(lock_file: BinaryIO) -> None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# File I/O layer
# ---------------------------------------------------------------------------


class TaskboardStore:
    """File-backed claim store for ``task_plans/taskboard.json``.

    All mutating methods take an exclusive advisory lock for atomic
    read-modify-write.  Read-only methods (``list_claims``, ``check``) take a
    shared lock on POSIX so they never block each other (Windows lacks shared
    locks, so they serialise there — still correct, just less concurrent).

    After each successful mutating call the JSON is written back and
    ``taskboard.md`` is re-rendered in the same lock window.

    The lock backend is platform-selected (``fcntl`` on POSIX, ``msvcrt`` on
    Windows) by ``_acquire_lock`` / ``_release_lock``.
    """

    def __init__(self, json_path: Path, md_path: Path | None = None) -> None:
        self._json_path = json_path
        self._md_path = md_path or json_path.parent / "taskboard.md"
        # Ensure parent directory exists.
        self._json_path.parent.mkdir(parents=True, exist_ok=True)

    # -- private helpers ---------------------------------------------------

    def _read_state(self) -> State:
        """Read + parse the JSON store.

        - File absent → return empty state (normal first-run case).
        - File present but not valid JSON / not a dict → raise immediately with
          an actionable message.  We must not silently wipe the board on a
          write-back failure or a partial file (fast-fail, not silent data loss).
        """
        if not self._json_path.exists():
            return empty_state()
        raw = self._json_path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"taskboard store is corrupt (invalid JSON) at "
                f"{self._json_path}: {exc}.  "
                "Inspect the file manually and remove or restore it before retrying."
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"taskboard store has unexpected top-level type "
                f"({type(data).__name__!r}) at {self._json_path}.  "
                "Expected a JSON object.  Inspect the file manually."
            )
        return data

    def _write_state(self, state: State) -> None:
        """Write state atomically (``os.replace``) so a crash mid-write never
        leaves a partial / empty JSON file in place of a good one.

        The markdown view is a derived / human-readable artefact; it is written
        directly (non-atomic) since losing it is recoverable.
        """
        tmp = self._json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, self._json_path)  # atomic same-volume rename (POSIX + Win)
        self._md_path.write_text(render_markdown(state), encoding="utf-8")

    def _with_lock(self, fn: Any, exclusive: bool = True) -> Any:
        """Open the lock file, acquire the advisory lock, call
        fn(state) -> (new_state, result), write back if exclusive (mutating),
        release lock.

        ``fn`` signature: ``(state: State) -> (State, result)`` for mutating calls,
        or ``(state: State) -> result`` for read-only calls (exclusive=False).
        """
        lock_path = self._json_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "ab") as lf:
            _acquire_lock(lf, exclusive)
            try:
                state = self._read_state()
                if exclusive:
                    # Lazy TTL reclaim on every mutating call.
                    state, _ = reclaim_stale(state)
                    new_state, result = fn(state)
                    self._write_state(new_state)
                    return result
                else:
                    return fn(state)
            finally:
                _release_lock(lf)

    # -- public API --------------------------------------------------------

    def claim(
        self,
        owner: str,
        paths: list[str],
        task: str,
        mode: str = "write",
        identity: str | None = None,
    ) -> dict[str, Any]:
        """Attempt to claim (paths, mode) for owner.

        ``identity`` is the conflict identity supplied by the server (derived from
        ``CLAUDE_CODE_SESSION_ID``; ADR-0022); it is not a wire parameter.  When
        omitted it falls back to ``owner``.

        Returns ``{status, claim_id, conflicts}``.  Paths are normalised before
        storing.  Mode must be ``"read"`` or ``"write"``.
        """
        _validate_mode(mode)
        # Guard before iterating: a stringified list would silently char-split.
        if not isinstance(paths, list):
            raise TypeError(f"expected list of paths, got {type(paths).__name__!r}")
        if not paths:
            raise ValueError("paths must be non-empty")
        norm_paths = [normalize_path(p) for p in paths]
        eff_identity = identity if identity is not None else owner

        def _mutate(state: State) -> tuple[State, dict[str, Any]]:
            new_state, c = add_claim(
                state, owner, norm_paths, task, mode, identity=eff_identity
            )
            conflicts = [
                {"owner": g["owner"], "paths": g["paths"], "mode": g["mode"]}
                for g in compute_conflicts(state, norm_paths, mode, eff_identity)
            ]
            return new_state, {
                "status": c["status"],
                "claim_id": c["claim_id"],
                "conflicts": conflicts,
            }

        return self._with_lock(_mutate)

    def release(self, claim_id: str) -> dict[str, Any]:
        """Release a claim and auto-promote pending claims.

        Returns ``{released_id, promoted: [{claim_id, owner, paths}]}``.
        """

        def _mutate(state: State) -> tuple[State, dict[str, Any]]:
            new_state, promoted = release_claim(state, claim_id)
            return new_state, {
                "released_id": claim_id,
                "promoted": [
                    {
                        "claim_id": p["claim_id"],
                        "owner": p["owner"],
                        "paths": p["paths"],
                    }
                    for p in promoted
                ],
            }

        return self._with_lock(_mutate)

    def check(
        self,
        paths: list[str],
        mode: str = "write",
        identity: str | None = None,
    ) -> dict[str, Any]:
        """Dry-run conflict check — zero side effects, shared lock.

        ``identity`` is the server-derived conflict identity (ADR-0022); granted
        claims of the same identity are not reported as conflicts, so a check
        mirrors what a subsequent ``claim`` from the same caller would see.  When
        omitted it falls back to ``owner``-less behaviour by using a sentinel that
        matches no existing claim, surfacing every overlapping grant.

        Returns ``{conflicts: [{owner, paths, mode}]}``.
        """
        _validate_mode(mode)
        # Guard before iterating: a stringified list would silently char-split.
        if not isinstance(paths, list):
            raise TypeError(f"expected list of paths, got {type(paths).__name__!r}")
        if not paths:
            raise ValueError("paths must be non-empty")
        norm_paths = [normalize_path(p) for p in paths]
        # No owner is passed to check, so when the server cannot supply an identity
        # we use a sentinel that never equals a stored identity — check then reports
        # all overlaps, which is the conservative answer for a dry run.
        eff_identity = identity if identity is not None else _NO_IDENTITY_SENTINEL

        def _read(state: State) -> dict[str, Any]:
            conflicts = compute_conflicts(state, norm_paths, mode, eff_identity)
            return {
                "conflicts": [
                    {"owner": c["owner"], "paths": c["paths"], "mode": c["mode"]}
                    for c in conflicts
                ]
            }

        return self._with_lock(_read, exclusive=False)

    def list_claims(self) -> dict[str, Any]:
        """Return current active (granted) + pending queue + recent released.

        Zero side effects, shared lock.
        """

        def _read(state: State) -> dict[str, Any]:
            claims: list[Claim] = state.get("claims", [])
            released: list[Claim] = state.get("released", [])
            return {
                "active": [c for c in claims if c["status"] == "granted"],
                "pending": [c for c in claims if c["status"] == "pending"],
                "recent_released": list(reversed(released[-_HISTORY_KEEP:])),
            }

        return self._with_lock(_read, exclusive=False)

    def wait(self, claim_id: str, timeout_s: float = 5.0) -> dict[str, Any]:
        """Block-poll until the pending claim is granted or timeout expires.

        Polls every 0.5 s; ``timeout_s`` is capped at ``MAX_WAIT_TIMEOUT``.
        Returns ``{status}`` where status is ``"granted"`` or ``"timeout"``.
        Fast-fails if claim_id is unknown.
        """
        timeout_s = min(float(timeout_s), MAX_WAIT_TIMEOUT)
        deadline = time.monotonic() + timeout_s

        while True:
            # Read-only probe — shared lock.
            def _read(state: State) -> str | None:
                for c in state.get("claims", []):
                    if c["claim_id"] == claim_id:
                        return c["status"]
                # Not in active list: check released history.
                for c in state.get("released", []):
                    if c["claim_id"] == claim_id:
                        return "released"
                return None  # completely unknown

            status = self._with_lock(_read, exclusive=False)
            if status is None:
                raise ValueError(f"unknown claim_id: {claim_id!r}")
            if status in ("granted", "released"):
                return {"status": status}
            # Still pending — check timeout.
            if time.monotonic() >= deadline:
                return {"status": "timeout"}
            time.sleep(0.5)

    def touch(self, claim_id: str) -> dict[str, Any]:
        """Heartbeat: update ``touched`` to extend TTL.

        Returns ``{claim_id, touched}``.
        """

        def _mutate(state: State) -> tuple[State, dict[str, Any]]:
            new_state, c = touch_claim(state, claim_id)
            return new_state, {"claim_id": claim_id, "touched": c["touched"]}

        return self._with_lock(_mutate)

    def force_release(self, claim_id: str) -> dict[str, Any]:
        """Forcibly release any active claim (including stale ones not yet TTL-reclaimed).

        Returns the same structure as ``release``.
        Rejects already-released / unknown IDs with a clear ValueError.
        """

        # Re-use the release logic — but we need the claim to still be in "claims".
        # Raises ValueError for already-released or completely unknown IDs.
        def _mutate(state: State) -> tuple[State, dict[str, Any]]:
            # Check if it's in released history (already gone).
            for c in state.get("released", []):
                if c["claim_id"] == claim_id:
                    raise ValueError(
                        f"claim {claim_id!r} is already released; "
                        "nothing to force-release"
                    )
            new_state, promoted = release_claim(state, claim_id)
            return new_state, {
                "released_id": claim_id,
                "promoted": [
                    {
                        "claim_id": p["claim_id"],
                        "owner": p["owner"],
                        "paths": p["paths"],
                    }
                    for p in promoted
                ],
            }

        return self._with_lock(_mutate)
