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
from typing import Any, Literal, Optional

# Maximum number of entries kept in memory (ring buffer).
_MAX_ENTRIES = 1000
# Maximum characters kept for a single params/result repr in an activity line.
_MAX_FIELD_LEN = 500
# Maximum characters kept for a single agent prose block shown inline.
_MAX_PROSE_LEN = 2000

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
    kind: Literal[
        "activity",
        "feedback",
        "diagnostic",
        # Embedded-agent stream kinds (B0):
        "assistant",  # prose text from the claude child
        "tool_use",  # a tool call the child is making
        "tool_result",  # the tool's return value
        "system",  # init / session metadata
        "result",  # terminal result frame (cost, reason)
    ]
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
        # B0 embedded-agent flag: when True the _after_success activity tap
        # in RemoteControlAdapter is skipped (the stream-json transcript
        # already carries every tool call; recording twice would be noise).
        self._embedded_active: bool = False
        # session_id from the last system/init frame; persisted for future
        # --resume support (B1); not used in B0.
        self._session_id: str = ""
        # Dedup: last session_id for which a [session] line was appended.
        # claude re-sends system/init on every stdin turn with the *same*
        # session_id; only append a new line when the id actually changes.
        self._last_session_id: Optional[str] = None
        # Accumulated per-session cost estimate across all turns.
        # total_cost_usd from each result frame is per-turn (not cumulative),
        # so we sum them here to report a session total in [DONE] lines.
        # Reset on clear() so the transcript display stays consistent.
        self._session_cost_usd: float = 0.0

    # ------------------------------------------------------------------
    # Record methods — main-thread only
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Embedded-agent control
    # ------------------------------------------------------------------

    def set_embedded_active(self, active: bool) -> None:
        """Enable or disable the embedded-agent mode.

        When active, ``_after_success`` in ``RemoteControlAdapter`` skips
        recording activity entries — the stream-json transcript is the
        authoritative source. Main-thread only.
        """
        self._embedded_active = active

    def is_embedded_active(self) -> bool:
        """True while the embedded claude process is live."""
        return self._embedded_active

    def set_session_id(self, session_id: str) -> None:
        """Store the session_id from the child's ``system/init`` frame (B1 hook)."""
        self._session_id = session_id

    def get_session_id(self) -> str:
        """Return the most recently received session_id."""
        return self._session_id

    # ------------------------------------------------------------------
    # Embedded-agent transcript record methods (B0)
    # ------------------------------------------------------------------

    def record_assistant(self, text: str) -> None:
        """Append a prose text block from the claude child. Main-thread only."""
        if len(text) > _MAX_PROSE_LEN:
            text = text[:_MAX_PROSE_LEN] + "…"
        self._append(
            TranscriptEntry(kind="assistant", text=text, timestamp=time.time())
        )

    def record_tool_use(self, tool_name: str, input_summary: str = "") -> None:
        """Append a tool-call marker (name only). Main-thread only.

        The tool input payload is intentionally NOT shown — it is noise in the
        conversation view. ``input_summary`` is accepted for caller compatibility
        but not rendered.
        """
        del input_summary
        text = f"⏺ {tool_name}"
        self._append(TranscriptEntry(kind="tool_use", text=text, timestamp=time.time()))

    def record_tool_result(self, summary: str) -> None:
        """Append a tool-result entry. Main-thread only."""
        text = f"[result] {summary}"
        self._append(
            TranscriptEntry(kind="tool_result", text=text, timestamp=time.time())
        )

    def record_system(self, session_id: str) -> None:
        """Append a system-init entry and store the session_id. Main-thread only.

        Dedup: claude re-sends system/init on every stdin turn with the same
        session_id. Only one [session] line per unique id is appended to avoid
        transcript clutter; the internal session_id is always updated.
        """
        self.set_session_id(session_id)
        if session_id == self._last_session_id:
            return
        self._last_session_id = session_id
        text = f"[session] id={session_id}"
        self._append(TranscriptEntry(kind="system", text=text, timestamp=time.time()))

    def record_result(
        self,
        is_error: bool,
        result_text: str,
        total_cost_usd: float,
        terminal_reason: str,
    ) -> None:
        """Append a terminal result entry. Main-thread only.

        Accumulates total_cost_usd across turns into _session_cost_usd and
        reports the running session total rather than the per-turn value.
        Per-turn cost from claude is an estimate; the session total is marked
        as such with "(session est.)".
        """
        status = "ERROR" if is_error else "DONE"
        self._session_cost_usd += total_cost_usd
        session_cost_str = f"cost≈${self._session_cost_usd:.4f} (session est.)"
        parts = [f"[{status}]", terminal_reason, session_cost_str]
        # On success the result_text duplicates the assistant prose already shown
        # via the streamed ``assistant`` frame, so omit it; keep it only on error
        # where it may carry an error detail not surfaced elsewhere.
        if result_text and is_error:
            snippet = result_text[:200] + ("…" if len(result_text) > 200 else "")
            parts.append(snippet)
        text = " ".join(parts)
        self._append(TranscriptEntry(kind="result", text=text, timestamp=time.time()))

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

        ``severity`` ∈ {"error", "info"}.  Suppressed while the embedded agent
        is active — GUI-internal diagnostics (e.g. "Device connected") are not
        part of the agent conversation and only add noise to the transcript.
        """
        if self._embedded_active:
            return
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
        """Remove all entries. Main-thread only.

        Resets the session-cost accumulator and session-id dedup state so that
        the next conversation starts fresh in the display.
        """
        self._entries.clear()
        self._session_cost_usd = 0.0
        self._last_session_id = None
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
