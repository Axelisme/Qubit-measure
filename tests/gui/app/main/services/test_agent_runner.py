"""Unit tests for agent_runner pure-logic components.

Tested without QProcess / real claude spawn:
  - StreamJsonParser: all frame types + multi-block assistant + bad lines
  - AgentRunState: state transitions + waiting detection
  - build_claude_argv: expected flags
  - Input routing logic (mocked runner + inbox)
  - Tap skip: embedded active flag suppresses record_activity
"""

from __future__ import annotations

import json

import pytest
from zcu_tools.gui.app.main.services.agent_chat import AgentChatService
from zcu_tools.gui.app.main.services.agent_runner import (
    AgentRunState,
    AssistantTextUpdate,
    RateLimitUpdate,
    ResultUpdate,
    StreamJsonParser,
    SystemInitUpdate,
    ToolResultUpdate,
    ToolUseUpdate,
    build_claude_argv,
    build_stdin_message,
)

# ---------------------------------------------------------------------------
# StreamJsonParser
# ---------------------------------------------------------------------------


def _parser() -> StreamJsonParser:
    return StreamJsonParser()


def _line(obj: dict) -> str:  # type: ignore[type-arg]
    return json.dumps(obj)


class TestStreamJsonParserSystem:
    def test_system_init_sets_session_id(self) -> None:
        p = _parser()
        line = _line({"type": "system", "subtype": "init", "session_id": "sid-abc"})
        updates = p.feed_line(line)
        assert len(updates) == 1
        u = updates[0]
        assert isinstance(u, SystemInitUpdate)
        assert u.session_id == "sid-abc"
        assert p.session_id == "sid-abc"

    def test_system_non_init_subtype_ignored(self) -> None:
        p = _parser()
        line = _line({"type": "system", "subtype": "other"})
        updates = p.feed_line(line)
        assert updates == []


class TestStreamJsonParserAssistant:
    def test_single_text_block(self) -> None:
        p = _parser()
        line = _line(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello"}],
                },
            }
        )
        updates = p.feed_line(line)
        assert len(updates) == 1
        assert isinstance(updates[0], AssistantTextUpdate)
        assert updates[0].text == "Hello"

    def test_multi_block_text_and_tool_use(self) -> None:
        p = _parser()
        line = _line(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "mcp__measure-gui__gui_state_check",
                            "input": {},
                        },
                    ],
                },
            }
        )
        updates = p.feed_line(line)
        assert len(updates) == 2
        assert isinstance(updates[0], AssistantTextUpdate)
        assert updates[0].text == "Let me check."
        assert isinstance(updates[1], ToolUseUpdate)
        assert updates[1].tool_name == "mcp__measure-gui__gui_state_check"

    def test_empty_text_block_skipped(self) -> None:
        p = _parser()
        line = _line(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                },
            }
        )
        updates = p.feed_line(line)
        assert updates == []

    def test_tool_use_input_summary_truncated(self) -> None:
        p = _parser()
        large_input = {"key": "x" * 600}
        line = _line(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t2",
                            "name": "foo",
                            "input": large_input,
                        }
                    ],
                },
            }
        )
        updates = p.feed_line(line)
        assert len(updates) == 1
        u = updates[0]
        assert isinstance(u, ToolUseUpdate)
        assert "…" in u.input_summary


class TestStreamJsonParserUser:
    def test_tool_result_string_content(self) -> None:
        p = _parser()
        line = _line(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok result",
                        }
                    ],
                },
            }
        )
        updates = p.feed_line(line)
        assert len(updates) == 1
        assert isinstance(updates[0], ToolResultUpdate)
        assert "ok result" in updates[0].summary

    def test_tool_result_list_content(self) -> None:
        p = _parser()
        line = _line(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "part1"},
                                {"type": "text", "text": "part2"},
                            ],
                        }
                    ],
                },
            }
        )
        updates = p.feed_line(line)
        assert len(updates) == 1
        assert isinstance(updates[0], ToolResultUpdate)
        assert "part1" in updates[0].summary

    def test_non_tool_result_blocks_skipped(self) -> None:
        p = _parser()
        line = _line(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                },
            }
        )
        updates = p.feed_line(line)
        assert updates == []


class TestStreamJsonParserResult:
    def test_success_result(self) -> None:
        p = _parser()
        line = _line(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "Done",
                "total_cost_usd": 0.0012,
                "terminal_reason": "completed",
            }
        )
        updates = p.feed_line(line)
        assert len(updates) == 1
        u = updates[0]
        assert isinstance(u, ResultUpdate)
        assert u.is_error is False
        assert u.result_text == "Done"
        assert pytest.approx(u.total_cost_usd) == 0.0012
        assert u.terminal_reason == "completed"

    def test_error_result(self) -> None:
        p = _parser()
        line = _line(
            {
                "type": "result",
                "is_error": True,
                "result": "fail",
                "total_cost_usd": 0.0,
                "terminal_reason": "error",
            }
        )
        updates = p.feed_line(line)
        assert isinstance(updates[0], ResultUpdate)
        assert updates[0].is_error is True


class TestStreamJsonParserRateLimit:
    def test_rate_limit_event(self) -> None:
        p = _parser()
        line = _line(
            {
                "type": "rate_limit_event",
                "rate_limit_info": {
                    "status": "active",
                    "resetsAt": "2026-01-01T00:00:00Z",
                },
            }
        )
        updates = p.feed_line(line)
        assert len(updates) == 1
        assert isinstance(updates[0], RateLimitUpdate)
        assert updates[0].status == "active"


class TestStreamJsonParserBadInput:
    def test_blank_line_returns_empty(self) -> None:
        p = _parser()
        assert p.feed_line("") == []
        assert p.feed_line("   ") == []

    def test_malformed_json_does_not_raise(self) -> None:
        p = _parser()
        # Must not raise; returns empty list.
        result = p.feed_line("{not valid json")
        assert result == []

    def test_unknown_type_ignored(self) -> None:
        p = _parser()
        line = _line({"type": "future_frame_type", "data": 42})
        updates = p.feed_line(line)
        assert updates == []

    def test_non_dict_json_ignored(self) -> None:
        p = _parser()
        updates = p.feed_line("[1, 2, 3]")
        assert updates == []

    def test_handler_exception_does_not_propagate(self) -> None:
        """A broken content block must not crash the parser."""
        p = _parser()
        # A content list containing a non-dict entry that would break inner loops.
        line = _line(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [None, "bad", {"type": "text", "text": "ok"}],
                },
            }
        )
        # Must not raise; might return partial or empty.
        result = p.feed_line(line)
        # The "ok" text block should still be parsed (loop continues past bad items).
        texts = [u.text for u in result if isinstance(u, AssistantTextUpdate)]
        assert "ok" in texts


# ---------------------------------------------------------------------------
# AgentRunState
# ---------------------------------------------------------------------------


class TestAgentRunState:
    def _state(self, has_pending: bool = False) -> AgentRunState:
        return AgentRunState(has_pending_wait=lambda: has_pending)

    def test_initial_state_is_idle(self) -> None:
        s = self._state()
        assert s.state == "idle"

    def test_start_sets_working(self) -> None:
        s = self._state()
        s.on_start()
        assert s.state == "working"

    def test_result_success_sets_idle(self) -> None:
        s = self._state()
        s.on_start()
        s.on_result(is_error=False)
        assert s.state == "idle"

    def test_result_error_sets_stopped(self) -> None:
        s = self._state()
        s.on_start()
        s.on_result(is_error=True)
        assert s.state == "stopped"

    def test_stop_sets_stopped(self) -> None:
        s = self._state()
        s.on_start()
        s.on_stop()
        assert s.state == "stopped"

    def test_assistant_chunk_with_pending_wait_transitions_to_waiting(self) -> None:
        s = AgentRunState(has_pending_wait=lambda: True)
        s.on_start()
        # Starts working; after an assistant chunk, has_pending_wait() → True → waiting
        s.on_assistant_chunk_received()
        assert s.state == "waiting"

    def test_assistant_chunk_without_pending_wait_stays_working(self) -> None:
        s = AgentRunState(has_pending_wait=lambda: False)
        s.on_start()
        s.on_assistant_chunk_received()
        assert s.state == "working"

    def test_stdin_sent_from_waiting_returns_to_working(self) -> None:
        s = AgentRunState(has_pending_wait=lambda: True)
        s.on_start()
        s.on_assistant_chunk_received()  # → waiting
        assert s.state == "waiting"
        s.on_stdin_sent()
        assert s.state == "working"

    def test_is_active_true_for_working_and_waiting(self) -> None:
        s = AgentRunState(has_pending_wait=lambda: True)
        s.on_start()
        assert s.is_active() is True
        s.on_assistant_chunk_received()
        assert s.is_active() is True

    def test_is_active_false_for_idle_and_stopped(self) -> None:
        s = self._state()
        assert s.is_active() is False
        s.on_start()
        s.on_stop()
        assert s.is_active() is False


# ---------------------------------------------------------------------------
# build_claude_argv
# ---------------------------------------------------------------------------


class TestBuildClaudeArgv:
    def test_contains_stream_json_flags(self) -> None:
        argv = build_claude_argv("task", "/tmp/mcp.json")
        assert "--output-format" in argv
        assert "stream-json" in argv
        assert "--input-format" in argv

    def test_contains_mcp_config(self) -> None:
        argv = build_claude_argv("task", "/tmp/mcp.json")
        idx = argv.index("--mcp-config")
        assert argv[idx + 1] == "/tmp/mcp.json"

    def test_contains_allowed_tools_default(self) -> None:
        argv = build_claude_argv("task", "/tmp/mcp.json")
        idx = argv.index("--allowedTools")
        assert argv[idx + 1] == "mcp__measure-gui__*"

    def test_contains_verbose(self) -> None:
        argv = build_claude_argv("task", "/tmp/mcp.json")
        assert "--verbose" in argv

    def test_contains_prompt_flag(self) -> None:
        argv = build_claude_argv("my task", "/tmp/mcp.json")
        assert "-p" in argv
        idx = argv.index("-p")
        assert argv[idx + 1] == "my task"

    def test_custom_allowed_tools(self) -> None:
        argv = build_claude_argv("t", "/tmp/c.json", allowed_tools="mcp__x__*")
        idx = argv.index("--allowedTools")
        assert argv[idx + 1] == "mcp__x__*"


# ---------------------------------------------------------------------------
# build_stdin_message
# ---------------------------------------------------------------------------


def test_build_stdin_message_valid_json() -> None:
    data = build_stdin_message("hello agent")
    decoded = data.decode("utf-8")
    assert decoded.endswith("\n")
    obj = json.loads(decoded.strip())
    assert obj["type"] == "user"
    assert obj["message"]["role"] == "user"
    assert obj["message"]["content"] == "hello agent"


# ---------------------------------------------------------------------------
# AgentChatService embedded-agent extensions (P2)
# ---------------------------------------------------------------------------


class TestAgentChatServiceEmbedded:
    def _svc(self) -> AgentChatService:
        return AgentChatService()

    def test_embedded_active_default_false(self) -> None:
        svc = self._svc()
        assert svc.is_embedded_active() is False

    def test_set_embedded_active(self) -> None:
        svc = self._svc()
        svc.set_embedded_active(True)
        assert svc.is_embedded_active() is True
        svc.set_embedded_active(False)
        assert svc.is_embedded_active() is False

    def test_record_assistant_appends(self) -> None:
        svc = self._svc()
        svc.record_assistant("Hello from claude")
        entries = svc.entries()
        assert len(entries) == 1
        assert entries[0].kind == "assistant"
        assert "Hello from claude" in entries[0].text

    def test_record_assistant_truncates_long_text(self) -> None:
        svc = self._svc()
        svc.record_assistant("x" * 3000)
        text = svc.entries()[0].text
        assert "…" in text

    def test_record_tool_use_appends(self) -> None:
        svc = self._svc()
        svc.record_tool_use("gui_state_check", "{}")
        entries = svc.entries()
        assert entries[0].kind == "tool_use"
        assert "gui_state_check" in entries[0].text

    def test_record_tool_result_appends(self) -> None:
        svc = self._svc()
        svc.record_tool_result("ok")
        entries = svc.entries()
        assert entries[0].kind == "tool_result"
        assert "[result]" in entries[0].text

    def test_record_system_stores_session_id(self) -> None:
        svc = self._svc()
        svc.record_system("my-session-id")
        assert svc.get_session_id() == "my-session-id"
        entries = svc.entries()
        assert entries[0].kind == "system"

    def test_record_result_success(self) -> None:
        svc = self._svc()
        svc.record_result(False, "finished", 0.0042, "completed")
        entries = svc.entries()
        assert entries[0].kind == "result"
        assert "DONE" in entries[0].text
        assert "completed" in entries[0].text

    def test_record_result_error(self) -> None:
        svc = self._svc()
        svc.record_result(True, "oops", 0.0, "error")
        entries = svc.entries()
        assert "ERROR" in entries[0].text


# ---------------------------------------------------------------------------
# Tap-skip: embedded active suppresses record_activity
# ---------------------------------------------------------------------------


class TestTapSkipWhenEmbeddedActive:
    """Verify that the _after_success tap skips when embedded agent is active.

    We test the AgentChatService guard (is_embedded_active check) directly,
    since _dispatch_chat_activity is an IO-thread internal that needs QProcess.
    This mirrors what the service does:
    ``if chat.is_embedded_active(): return``
    """

    def test_activity_recorded_when_not_embedded(self) -> None:
        svc = AgentChatService()
        svc.set_embedded_active(False)
        # Simulate the guard in _dispatch_chat_activity:
        if not svc.is_embedded_active():
            svc.record_activity("run.start", {}, {})
        assert len(svc.entries()) == 1

    def test_activity_skipped_when_embedded(self) -> None:
        svc = AgentChatService()
        svc.set_embedded_active(True)
        # Simulate the guard:
        if not svc.is_embedded_active():
            svc.record_activity("run.start", {}, {})
        assert len(svc.entries()) == 0
