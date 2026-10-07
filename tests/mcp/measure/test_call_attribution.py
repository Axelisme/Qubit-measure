"""Native MCP identity scope observed through public measure session methods.

Ledger forwarding and continuation attribution are separate #24 consumers;
these cases cover only the synchronous native scope producer.
"""

import io
import json
import sys
from collections.abc import Iterator
from contextvars import Context
from pathlib import Path

import pytest
from zcu_tools.mcp.core.bridge import MCPBridgeConfig
from zcu_tools.mcp.core.stdio_server import (
    McpServerConfig,
    StdioLoopHooks,
    ToolTable,
    run_stdio_loop,
)
from zcu_tools.mcp.measure.session import MeasureMcpSession


@pytest.fixture
def sessions(tmp_path: Path) -> Iterator[tuple[MeasureMcpSession, MeasureMcpSession]]:
    config = MCPBridgeConfig(
        tool_prefix="",
        server_display_name="call-scope-test",
        server_instructions="",
        app_name="gui",
        default_port=8765,
        mcp_version=75,
        wire_version=1,
        pid_file=tmp_path / "unused.pid",
        log_file=tmp_path / "unused.log",
        run_script_name="run_measure_gui.py",
    )
    pair = tuple(
        MeasureMcpSession(
            config,
            recipes=(),
            resolve_connect_port=lambda config, requested: (
                config.default_port if requested is None else requested
            ),
            port_is_open=lambda _port: False,
        )
        for _ in range(2)
    )
    first, second = pair
    try:
        yield first, second
    finally:
        first.close()
        second.close()


def test_native_scope_restores_nested_and_exceptional_calls(
    sessions: tuple[MeasureMcpSession, MeasureMcpSession],
) -> None:
    session, _ = sessions
    with pytest.raises(RuntimeError):
        session.current_call_id()

    with session.tool_call_scope("  outer native  "):
        assert session.current_call_id() == "  outer native  "
        with session.tool_call_scope("inner"):
            assert session.current_call_id() == "inner"
        assert session.current_call_id() == "  outer native  "
        with (
            pytest.raises(ValueError, match="handler failed"),
            session.tool_call_scope("failed"),
        ):
            raise ValueError("handler failed")
        assert session.current_call_id() == "  outer native  "

    with pytest.raises(RuntimeError):
        session.current_call_id()


def test_native_scope_is_local_to_session_and_execution_context(
    sessions: tuple[MeasureMcpSession, MeasureMcpSession],
) -> None:
    first, second = sessions
    with first.tool_call_scope("first"):
        with pytest.raises(RuntimeError):
            second.current_call_id()
        with pytest.raises(RuntimeError):
            Context().run(first.current_call_id)
        with second.tool_call_scope("second"):
            assert first.current_call_id() == "first"
            assert second.current_call_id() == "second"
        assert first.current_call_id() == "first"
        with pytest.raises(RuntimeError):
            second.current_call_id()
    with pytest.raises(RuntimeError):
        first.current_call_id()


@pytest.mark.parametrize("native_id", ["request-native", -31])
def test_stdio_handler_observes_real_measure_session_native_scope(
    sessions: tuple[MeasureMcpSession, MeasureMcpSession],
    monkeypatch: pytest.MonkeyPatch,
    native_id: str | int,
) -> None:
    session, _ = sessions
    request = {
        "jsonrpc": "2.0",
        "id": native_id,
        "method": "tools/call",
        "params": {"name": "observe"},
    }
    output = io.BytesIO()
    monkeypatch.setattr(
        sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO((json.dumps(request) + "\n").encode())),
    )
    monkeypatch.setattr(
        sys,
        "stdout",
        io.TextIOWrapper(output, write_through=True),
    )

    def observe(_arguments: dict[str, object]) -> str:
        return session.current_call_id()

    tools: ToolTable = {
        "observe": {"handler": observe, "description": "", "inputSchema": {}}
    }
    run_stdio_loop(
        McpServerConfig("", "scope-test", ""),
        tools,
        hooks=StdioLoopHooks(tool_call_scope=session.tool_call_scope),
    )

    reply = json.loads(output.getvalue())
    expected = native_id if isinstance(native_id, str) else str(native_id)
    assert reply["result"]["content"] == [{"type": "text", "text": expected}]
    with pytest.raises(RuntimeError):
        session.current_call_id()


@pytest.mark.parametrize("call_id", ["", " ", "\t\n"])
def test_blank_native_id_does_not_replace_existing_scope(
    sessions: tuple[MeasureMcpSession, MeasureMcpSession], call_id: str
) -> None:
    session, _ = sessions
    with session.tool_call_scope("valid"):
        with pytest.raises(ValueError, match=r"\S"), session.tool_call_scope(call_id):
            pytest.fail("blank call ID entered the handler")
        assert session.current_call_id() == "valid"
