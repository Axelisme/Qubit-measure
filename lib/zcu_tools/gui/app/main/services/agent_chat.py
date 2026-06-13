"""AgentChatService — owns the agent conversation transcript.

A pure Python service (no Qt dependency) that records three kinds of entries:
  - "activity": MCP tool calls routed through RemoteControlAdapter._after_success
  - "feedback": user text sent via the feedback inbox
  - "diagnostic": Controller diagnostics (error/info) fanned out to this service

The service holds a ring buffer (~1000 entries, dropping oldest) and a plain
observer list (not QObject) that the AgentChatDialog registers into. Main-thread
only for all mutation methods (docstring + callers' responsibility).

ADR-0023: feedback goes through FeedbackInbox → OperationHandles await path;
this service is a *display* layer only and never touches OperationHandles.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

# Maximum number of entries kept in memory (ring buffer).
_MAX_ENTRIES = 1000
# Maximum characters kept for a single params/result repr in an activity line.
_MAX_FIELD_LEN = 500

# Methods considered pure queries — never recorded as activity entries.
# Heuristic: keep list explicit and conservative (better to miss a recording
# than to flood with poll/snapshot noise).
_SKIP_METHODS: frozenset[str] = frozenset(
    {
        # polling / waiting
        "operation.await",
        "operation.poll",
        "operation.progress",
        "run.poll",
        "analyze.poll",
        "post_analyze.poll",
        "device.poll",
        "connect.poll",
        # pure reads
        "resources.versions",
        "state.check",
        "tab.list",
        "tab.get_cfg",
        "tab.get_cfg_summary",
        "tab.get_analyze_params",
        "tab.get_analyze_result",
        "tab.get_post_analyze_params",
        "tab.get_post_analyze_result",
        "tab.get_current_figure",
        "tab.list_paths",
        "tab.snapshot",
        "adapter.list",
        "adapter.cfg_spec",
        "adapter.analyze_spec",
        "adapter.guide",
        "soc.info",
        "device.list",
        "device.snapshot",
        "device.active_setup",
        "device.active_operation",
        "context.active",
        "context.labels",
        "context.get_md",
        "context.get_md_attr",
        "context.get_ml",
        "ml.list_roles",
        "predictor.info",
        "run.running_tab",
        "dialog.list_open",
        "dialog.screenshot",
        "view.screenshot",
        "view.snapshot",
        "editor.get",
        "editor.subscribe",
        "editor.unsubscribe",
        "save.set_paths",
        "startup.apply",
        # connect.start is a command; connect.wait is a long-wait (skip as await)
        "connect.wait",
        "run.wait",
        "analyze.wait",
        "post_analyze.wait",
        "device.wait",
    }
)


def _should_record(method: str) -> bool:
    """True when ``method`` is a side-effecting command worth showing in transcript.

    Conservative: anything not in the skip list and not matching common
    read-only suffixes is recorded. Prefix matching catches future poll/* expansions.
    """
    if method in _SKIP_METHODS:
        return False
    # Structural read prefixes that expand the skip list heuristically.
    _READ_PREFIXES = ("resources.", "state.")
    return not any(method.startswith(pfx) for pfx in _READ_PREFIXES)


def _short(obj: Any) -> str:
    """Compact, truncated repr of a params dict or result value."""
    text = repr(obj)
    if len(text) > _MAX_FIELD_LEN:
        return text[:_MAX_FIELD_LEN] + "…"
    return text


@dataclass(frozen=True)
class TranscriptEntry:
    kind: Literal["activity", "feedback", "diagnostic"]
    text: str
    timestamp: float  # time.time()


class AgentChatService:
    """Owns the agent conversation transcript (ring buffer + observer list).

    All mutation methods (record_*, clear) must be called on the Qt main thread.
    Observers are notified synchronously on the calling thread immediately after
    each append — callers must not hold locks that observers would need.
    """

    def __init__(self) -> None:
        self._entries: deque[TranscriptEntry] = deque(maxlen=_MAX_ENTRIES)
        self._listeners: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Record methods — main-thread only
    # ------------------------------------------------------------------

    def record_activity(
        self,
        method: str,
        params: Any,
        result: Any,
    ) -> None:
        """Append one agent tool-call activity line. Main-thread only.

        Formats as ``→ {method}({short_params}) ⇒ {short_result}``.
        Skips if _should_record returns False (pure queries / polls).
        """
        if not _should_record(method):
            return
        text = f"→ {method}({_short(params)}) ⇒ {_short(result)}"
        self._append(TranscriptEntry(kind="activity", text=text, timestamp=time.time()))

    def record_feedback(self, text: str) -> None:
        """Append a user feedback line. Main-thread only."""
        entry = TranscriptEntry(
            kind="feedback",
            text=f"you: {text}",
            timestamp=time.time(),
        )
        self._append(entry)

    def record_diagnostic(self, severity: str, title: str, message: str) -> None:
        """Append a Controller diagnostic. Main-thread only.

        ``severity`` ∈ {"error", "info"}.  The format mirrors the wire's
        diagnostic push so agents can correlate.
        """
        label = severity.upper()
        parts = [f"[{label}]"]
        if title:
            parts.append(title)
        if message:
            parts.append(message)
        text = " — ".join(parts)
        self._append(
            TranscriptEntry(kind="diagnostic", text=text, timestamp=time.time())
        )

    def clear(self) -> None:
        """Remove all entries. Main-thread only."""
        self._entries.clear()
        self._notify_listeners()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def entries(self) -> tuple[TranscriptEntry, ...]:
        """Return an immutable snapshot of the current transcript."""
        return tuple(self._entries)

    # ------------------------------------------------------------------
    # Observer management — safe to call from any thread (list ops are GIL-safe
    # enough for add/remove; notification is synchronous on the calling thread).
    # ------------------------------------------------------------------

    def add_listener(self, cb: Callable[[], None]) -> None:
        if cb not in self._listeners:
            self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[], None]) -> None:
        try:
            self._listeners.remove(cb)
        except ValueError:
            pass  # idempotent remove — already gone

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append(self, entry: TranscriptEntry) -> None:
        self._entries.append(entry)
        self._notify_listeners()

    def _notify_listeners(self) -> None:
        for cb in list(self._listeners):
            try:
                cb()
            except Exception:
                import logging

                logging.getLogger(__name__).exception(
                    "AgentChatService listener raised; ignoring"
                )
