"""AgentRunner — spawn a child ``claude`` process and stream its transcript.

B0 embedded-agent feature: a single child ``claude`` process is spawned against
the loopback measure-gui MCP server (which connects back to the already-running
GUI via session discovery). The child's stdout is a stream-json line protocol;
each line is parsed and routed into ``AgentChatService`` as transcript entries.

Threading model
---------------
- ``AgentRunner`` is a QObject; all public methods and signal emissions are
  main-thread only.
- ``QProcess`` delivers ``readyReadStandardOutput`` on the main thread via
  Qt's internal socket notifier. The ``_on_stdout`` slot is therefore also on
  the main thread, satisfying ``AgentChatService``'s main-thread invariant.

Loopback MCP config
-------------------
``build_loopback_mcp_config`` writes a temp ``.mcp.json`` that points the
spawned claude at the already-running GUI's MCP server entry point (same
``uv run`` command used in the repo's ``.mcp.json``). The spawned server
will discover the live GUI's port via session-discovery
(``~/.cache/zcu-tools/sessions/measure.json``).

The config is written to ``tempfile.gettempdir()/zcu_agent_mcp.json`` — a
single fixed path per machine, overwritten on each spawn. B0 only supports
one embedded agent at a time, so no per-session naming is needed.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from collections.abc import Callable

from qtpy.QtCore import (  # type: ignore[attr-defined]
    QObject,
    QProcess,
    QProcessEnvironment,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stream-json schema constants
# ---------------------------------------------------------------------------

# ``type`` values produced by ``claude --output-format stream-json``
_TYPE_SYSTEM = "system"
_TYPE_ASSISTANT = "assistant"
_TYPE_USER = "user"
_TYPE_RESULT = "result"
_TYPE_RATE_LIMIT = "rate_limit_event"

# Content-block type values inside ``message.content[]``
_CONTENT_TEXT = "text"
_CONTENT_TOOL_USE = "tool_use"
_CONTENT_TOOL_RESULT = "tool_result"

# Subtype for the init frame
_SUBTYPE_INIT = "init"

# ---------------------------------------------------------------------------
# Pure data types — TranscriptUpdate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssistantTextUpdate:
    """A prose text block from the assistant."""

    text: str


@dataclass(frozen=True)
class ToolUseUpdate:
    """A tool-call block from the assistant (before the tool executes)."""

    tool_name: str
    input_summary: str  # truncated repr of input dict


@dataclass(frozen=True)
class ToolResultUpdate:
    """A tool-result block (the user turn carrying tool output)."""

    summary: str


@dataclass(frozen=True)
class SystemInitUpdate:
    """The initial ``system/init`` frame carrying the session_id."""

    session_id: str


@dataclass(frozen=True)
class ResultUpdate:
    """The terminal ``result`` frame."""

    is_error: bool
    result_text: str
    total_cost_usd: float
    terminal_reason: str


@dataclass(frozen=True)
class RateLimitUpdate:
    """A rate-limit event (informational)."""

    status: str


# Union of all transcript updates the parser can emit.
TranscriptUpdate = (
    AssistantTextUpdate
    | ToolUseUpdate
    | ToolResultUpdate
    | SystemInitUpdate
    | ResultUpdate
    | RateLimitUpdate
)

# ---------------------------------------------------------------------------
# AgentRunState
# ---------------------------------------------------------------------------

AgentState = Literal["idle", "working", "waiting", "stopped"]


class AgentRunState:
    """Lightweight state machine for the embedded agent lifecycle.

    Transitions:
      idle    → working   on start()
      working → waiting   when has_pending_wait() is True (checked after
                          each assistant turn or tool-use batch finishes)
      working → idle      on result (terminal_reason=="completed")
      working → stopped   on stop() or result (is_error / other reason)
      waiting → working   when an input message is sent via stdin
      waiting → stopped   on stop()
      stopped → idle      on reset() (new start clears old state)

    The ``waiting`` state is informational for the dialog's input routing
    (feedback vs stdin message). The actual has_pending_wait() comes from the
    Controller's OperationHandles — we call a supplied callable each time we
    need to know.
    """

    def __init__(self, has_pending_wait: Callable[[], bool]) -> None:
        self._has_pending_wait = has_pending_wait
        self._state: AgentState = "idle"

    @property
    def state(self) -> AgentState:
        return self._state

    def on_start(self) -> None:
        """Called when claude process is launched."""
        self._state = "working"

    def on_stdin_sent(self) -> None:
        """Called after sending a message via stdin."""
        # If we were waiting, sending moves us back to working.
        if self._state in ("waiting", "working"):
            self._state = "working"

    def on_assistant_chunk_received(self) -> None:
        """Called after processing an assistant message frame.

        Re-evaluates whether we are now in 'waiting' state (i.e. an operation
        is live and the agent is waiting for a feedback wakeup).
        """
        if self._state == "working" and self._has_pending_wait():
            self._state = "waiting"

    def on_result(self, is_error: bool) -> None:
        """Called when the terminal ``result`` frame arrives."""
        if is_error:
            self._state = "stopped"
        else:
            self._state = "idle"

    def on_stop(self) -> None:
        """Called when stop() is issued externally."""
        self._state = "stopped"

    def is_active(self) -> bool:
        """True when a claude process is live (working or waiting)."""
        return self._state in ("working", "waiting")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

_MAX_INPUT_LEN = 300  # chars to show in tool_use input summary


def _truncate(text: str, max_len: int = _MAX_INPUT_LEN) -> str:
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def build_claude_argv(
    task: str,
    mcp_config_path: str,
    allowed_tools: str = "mcp__measure-gui__*",
) -> list[str]:
    """Build the argv list for the embedded ``claude`` child process.

    Uses ``--output-format stream-json --input-format stream-json`` so the
    parent can parse each stdout line as a complete JSON object and pipe
    additional user messages through stdin.

    ``--verbose`` is included so the stream-json transcript contains
    tool-use / tool-result frames (without it the transcript is prose-only).

    ``--allowedTools`` restricts tool access to the measure-gui MCP server
    to prevent unintended side-effects on other tools.

    B0: we pass ``-p <task>`` to seed the first turn, but keep stdin open so
    ``send_user_message`` can inject follow-up messages.  ``claude`` in
    interactive stream-json mode reads additional user turns from stdin after
    the initial ``-p`` turn completes.
    """
    return [
        "claude",
        "--output-format",
        "stream-json",
        "--input-format",
        "stream-json",
        "--verbose",
        "--mcp-config",
        mcp_config_path,
        "--allowedTools",
        allowed_tools,
        "-p",
        task,
    ]


def build_loopback_mcp_config(repo_root: str) -> str:
    """Write a temp ``.mcp.json`` that points claude at the measure-gui MCP server.

    The server entry mirrors the repo's top-level ``.mcp.json``: it uses
    ``uv run --extra gui python lib/zcu_tools/mcp/measure/server.py`` with the
    repo root as cwd. When the spawned MCP server starts, it reads the session
    discovery file (``~/.cache/zcu-tools/sessions/measure.json``) to locate
    the already-running GUI's TCP port.

    Returns the absolute path to the written config file.
    """
    config: dict[str, object] = {
        "mcpServers": {
            "measure-gui": {
                "type": "stdio",
                "command": "uv",
                "args": [
                    "run",
                    "--extra",
                    "gui",
                    "python",
                    "lib/zcu_tools/mcp/measure/server.py",
                ],
                # cwd must be the repo root so uv finds pyproject.toml and the
                # relative script path resolves correctly.
                "cwd": repo_root,
                "env": {},
            }
        }
    }
    path = Path(tempfile.gettempdir()) / "zcu_agent_mcp.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# StreamJsonParser
# ---------------------------------------------------------------------------


class StreamJsonParser:
    """Parse ``claude --output-format stream-json`` stdout lines.

    Each line must be a complete JSON object.  The parser produces a list of
    ``TranscriptUpdate`` for each line fed via ``feed_line``.  Malformed lines
    are logged and skipped — the parser never raises.

    Maintains ``session_id`` as mutable state because it arrives in the first
    ``system/init`` frame and is referenced by subsequent frames.
    """

    def __init__(self) -> None:
        self.session_id: str = ""

    def feed_line(self, raw: str) -> list[TranscriptUpdate]:
        """Parse one raw stdout line and return zero or more ``TranscriptUpdate``s."""
        raw = raw.strip()
        if not raw:
            return []
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("stream-json: malformed line ignored: %.120r", raw)
            return []
        if not isinstance(obj, dict):
            logger.debug("stream-json: non-object line ignored: %.60r", raw)
            return []

        frame_type = obj.get("type", "")
        try:
            return self._dispatch(frame_type, obj)
        except Exception:
            logger.exception(
                "stream-json: error handling frame type=%r; line ignored", frame_type
            )
            return []

    def _dispatch(self, frame_type: str, obj: dict) -> list[TranscriptUpdate]:  # type: ignore[type-arg]
        if frame_type == _TYPE_SYSTEM:
            return self._handle_system(obj)
        if frame_type == _TYPE_ASSISTANT:
            return self._handle_assistant(obj)
        if frame_type == _TYPE_USER:
            return self._handle_user(obj)
        if frame_type == _TYPE_RESULT:
            return self._handle_result(obj)
        if frame_type == _TYPE_RATE_LIMIT:
            return self._handle_rate_limit(obj)
        # Unknown frame type — log at debug and skip.
        logger.debug("stream-json: unknown frame type=%r", frame_type)
        return []

    def _handle_system(self, obj: dict) -> list[TranscriptUpdate]:  # type: ignore[type-arg]
        if obj.get("subtype") != _SUBTYPE_INIT:
            return []
        session_id = str(obj.get("session_id", ""))
        self.session_id = session_id
        return [SystemInitUpdate(session_id=session_id)]

    def _handle_assistant(self, obj: dict) -> list[TranscriptUpdate]:  # type: ignore[type-arg]
        message = obj.get("message", {})
        if not isinstance(message, dict):
            return []
        content = message.get("content", [])
        if not isinstance(content, list):
            return []
        updates: list[TranscriptUpdate] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == _CONTENT_TEXT:
                text = str(block.get("text", ""))
                if text:
                    updates.append(AssistantTextUpdate(text=text))
            elif block_type == _CONTENT_TOOL_USE:
                name = str(block.get("name", ""))
                input_obj = block.get("input", {})
                input_summary = _truncate(json.dumps(input_obj, ensure_ascii=False))
                updates.append(
                    ToolUseUpdate(tool_name=name, input_summary=input_summary)
                )
        return updates

    def _handle_user(self, obj: dict) -> list[TranscriptUpdate]:  # type: ignore[type-arg]
        """User frames carry tool_result blocks (the agent's tool output)."""
        message = obj.get("message", {})
        if not isinstance(message, dict):
            return []
        content = message.get("content", [])
        if not isinstance(content, list):
            return []
        updates: list[TranscriptUpdate] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != _CONTENT_TOOL_RESULT:
                continue
            result_content = block.get("content", "")
            # content may be a string or a list of blocks — normalise to a string.
            if isinstance(result_content, list):
                texts = [
                    b.get("text", "") for b in result_content if isinstance(b, dict)
                ]
                summary = _truncate(" ".join(texts))
            else:
                summary = _truncate(str(result_content))
            updates.append(ToolResultUpdate(summary=summary))
        return updates

    def _handle_result(self, obj: dict) -> list[TranscriptUpdate]:  # type: ignore[type-arg]
        is_error = bool(obj.get("is_error", False))
        result_text = str(obj.get("result", ""))
        total_cost = float(obj.get("total_cost_usd") or 0.0)
        terminal_reason = str(obj.get("terminal_reason", ""))
        return [
            ResultUpdate(
                is_error=is_error,
                result_text=result_text,
                total_cost_usd=total_cost,
                terminal_reason=terminal_reason,
            )
        ]

    def _handle_rate_limit(self, obj: dict) -> list[TranscriptUpdate]:  # type: ignore[type-arg]
        info = obj.get("rate_limit_info", {})
        status = str(info.get("status", "")) if isinstance(info, dict) else ""
        return [RateLimitUpdate(status=status)]


# ---------------------------------------------------------------------------
# Input envelope builder
# ---------------------------------------------------------------------------


def build_stdin_message(text: str) -> bytes:
    """Encode a user message as a stream-json stdin line.

    Format: ``{"type":"user","message":{"role":"user","content":"<text>"}}\n``
    """
    envelope = {
        "type": "user",
        "message": {"role": "user", "content": text},
    }
    return (json.dumps(envelope, ensure_ascii=False) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# AgentRunner (QObject, main-thread)
# ---------------------------------------------------------------------------

# Maximum bytes to buffer from stdout before discarding partial lines.
# QProcess reads until readyReadStandardOutput; we accumulate into a bytearray.
_STDOUT_BUF_MAX = 10 * 1024 * 1024  # 10 MiB guard


@dataclass
class _RunnerCallbacks:
    """Typed callback bundle injected by the caller (dialog or controller)."""

    on_update: Callable[[list[TranscriptUpdate]], None]
    """Called on main thread with a batch of TranscriptUpdates."""

    on_state_changed: Callable[[AgentState], None]
    """Called on main thread whenever AgentRunState.state changes."""

    on_process_error: Callable[[str], None]
    """Called on main thread when QProcess emits an error (not a result frame)."""

    has_pending_wait: Callable[[], bool]
    """Forwarded to AgentRunState for state-machine transitions."""


class AgentRunner(QObject):
    """Manage a child ``claude`` process and route its stdout to the transcript.

    Lifecycle:
      1. ``start(task, repo_root)`` — spawn the process.
      2. ``send_user_message(text)`` — write a stdin line.
      3. ``stop()`` — send SIGINT; the process should exit gracefully.
      4. After the terminal ``result`` frame, the process exits on its own.

    All methods are main-thread only (QObject rule).

    ``callbacks`` is injected at construction; the dialog owns the runner.
    """

    def __init__(
        self, callbacks: _RunnerCallbacks, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._callbacks = callbacks
        self._parser = StreamJsonParser()
        self._run_state = AgentRunState(callbacks.has_pending_wait)
        self._process: QProcess | None = None
        self._stdout_buf = bytearray()
        # Path to the temp MCP config written by the last start() call.
        self._mcp_config_path: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        return self._run_state.state

    def start(self, task: str, repo_root: str) -> None:
        """Spawn ``claude`` with the loopback MCP config.

        Fast-fails if a process is already running.
        """
        if (
            self._process is not None
            and self._process.state() != QProcess.ProcessState.NotRunning
        ):
            logger.warning(
                "AgentRunner.start() called while process still running; ignored"
            )
            return

        self._parser = StreamJsonParser()
        self._stdout_buf = bytearray()
        self._run_state.on_start()

        self._mcp_config_path = build_loopback_mcp_config(repo_root)
        argv = build_claude_argv(task, self._mcp_config_path)

        proc = QProcess(self)
        # Inherit the current environment; do NOT set ANTHROPIC_API_KEY —
        # the subscription auth is handled by claude itself.
        env = QProcessEnvironment.systemEnvironment()
        # Explicitly unset ANTHROPIC_API_KEY so claude uses subscription auth.
        env.remove("ANTHROPIC_API_KEY")
        proc.setProcessEnvironment(env)

        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.readyReadStandardError.connect(self._on_stderr)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)

        program = argv[0]
        args = argv[1:]
        proc.start(program, args)
        self._process = proc

        self._emit_state()
        logger.info("AgentRunner: spawned claude pid=%s", proc.processId())

    def send_user_message(self, text: str) -> None:
        """Write a user message to claude's stdin.

        Main-thread only.  No-op if no process is running.
        """
        proc = self._process
        if proc is None or proc.state() == QProcess.ProcessState.NotRunning:
            logger.warning("AgentRunner.send_user_message(): no live process; ignored")
            return
        data = build_stdin_message(text)
        proc.write(data)
        self._run_state.on_stdin_sent()
        self._emit_state()

    def stop(self) -> None:
        """Send SIGINT to the child process for a graceful shutdown.

        Main-thread only.
        """
        proc = self._process
        if proc is None or proc.state() == QProcess.ProcessState.NotRunning:
            return
        pid = proc.processId()
        if pid > 0:
            try:
                os.kill(pid, signal.SIGINT)
                logger.info("AgentRunner: sent SIGINT to pid=%s", pid)
            except OSError:
                logger.warning(
                    "AgentRunner: os.kill SIGINT failed for pid=%s", pid, exc_info=True
                )
                # Fall back to SIGTERM via Qt if SIGINT failed.
                proc.terminate()
        else:
            proc.terminate()
        self._run_state.on_stop()
        self._emit_state()

    def session_id(self) -> str:
        """Return the session_id from the last ``system/init`` frame (B1 use)."""
        return self._parser.session_id

    # ------------------------------------------------------------------
    # QProcess slots (main-thread)
    # ------------------------------------------------------------------

    def _on_stdout(self) -> None:
        proc = self._process
        if proc is None:
            return
        chunk = proc.readAllStandardOutput()
        if not chunk:
            return
        # QByteArray.data() → bytes → bytearray accumulation.
        self._stdout_buf.extend(chunk.data())
        if len(self._stdout_buf) > _STDOUT_BUF_MAX:
            logger.warning("AgentRunner: stdout buffer overflow; clearing")
            self._stdout_buf.clear()
            return
        self._flush_lines()

    def _flush_lines(self) -> None:
        """Split buffered bytes on newlines and feed complete lines to the parser."""
        while True:
            nl = self._stdout_buf.find(b"\n")
            if nl == -1:
                break
            line_bytes = self._stdout_buf[:nl]
            del self._stdout_buf[: nl + 1]
            line = line_bytes.decode("utf-8", errors="replace")
            updates = self._parser.feed_line(line)
            if updates:
                self._route_updates(updates)

    def _route_updates(self, updates: list[TranscriptUpdate]) -> None:
        """Route parsed updates to AgentChatService and state machine."""
        self._callbacks.on_update(updates)
        has_assistant = any(
            isinstance(u, (AssistantTextUpdate, ToolUseUpdate)) for u in updates
        )
        if has_assistant:
            self._run_state.on_assistant_chunk_received()
        for update in updates:
            if isinstance(update, ResultUpdate):
                self._run_state.on_result(update.is_error)
        self._emit_state()

    def _on_stderr(self) -> None:
        proc = self._process
        if proc is None:
            return
        data = proc.readAllStandardError().data()
        if data:
            logger.debug(
                "claude stderr: %s", data.decode("utf-8", errors="replace").strip()
            )

    def _on_process_finished(
        self, exit_code: int, exit_status: QProcess.ExitStatus
    ) -> None:
        logger.info(
            "AgentRunner: process finished exit_code=%s exit_status=%s",
            exit_code,
            exit_status,
        )
        # Flush any remaining buffered output.
        self._flush_lines()
        # If no terminal ResultUpdate was parsed (e.g. process killed), force stopped.
        if self._run_state.state not in ("idle", "stopped"):
            self._run_state.on_stop()
            self._emit_state()

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        msg = f"QProcess error: {error}"
        logger.error("AgentRunner: %s", msg)
        self._run_state.on_stop()
        self._emit_state()
        self._callbacks.on_process_error(msg)

    def _emit_state(self) -> None:
        self._callbacks.on_state_changed(self._run_state.state)
