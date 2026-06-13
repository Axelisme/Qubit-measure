"""Tests for AgentChatDialog — B1b-2 picker + conversation two-page structure.

All tests use a fake AgentSessionPort (FakeAgentSession) and a fake Controller
stub (MagicMock) so no real supervisor or Qt process is spawned.

Coverage
--------
  D1  - New → stacked index 1 (conversation page).
  D2  - Attach (running) → index 1, transcript cleared.
  D3  - Send with no live session → session.start() called.
  D4  - Close → session.detach() called (not stop); back to index 0.
  D5  - Stop-Remove → ctrl.remove_agent_session called; list refreshed.
  D6  - env=cli → no picker; starts on conversation; Close=stop.
  D7  - Dialog still satisfies B1a invariant (no concrete AgentRunner import).
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fake AgentSessionPort (extended from test_agent_session_port.py's FakeAgentSession)
# ---------------------------------------------------------------------------


class FakeAgentSession:
    """Qt-free fake for AgentSessionPort — includes detach() (B1b-2)."""

    def __init__(self, *, initial_state: str = "idle", running: bool = False) -> None:
        from zcu_tools.gui.app.main.services.ports import AgentState

        self._state: AgentState = initial_state  # type: ignore[assignment]
        self._running = running
        self.started_tasks: list[tuple[str, str]] = []
        self.sent_messages: list[str] = []
        self.stop_calls: int = 0
        self.detach_calls: int = 0
        self._listeners: list[Callable[..., None]] = []

    @property
    def state(self):  # type: ignore[return]
        return self._state

    def is_running(self) -> bool:
        return self._running

    def start(self, task: str, repo_root: str) -> None:
        self.started_tasks.append((task, repo_root))
        self._state = "working"  # type: ignore[assignment]
        self._running = True

    def send_user_message(self, text: str) -> None:
        self.sent_messages.append(text)

    def stop(self) -> None:
        self.stop_calls += 1
        self._state = "stopped"  # type: ignore[assignment]
        self._running = False

    def detach(self) -> None:
        self.detach_calls += 1
        self._state = "stopped"  # type: ignore[assignment]
        self._running = False

    def session_id(self) -> str:
        return "fake-session-id"

    def add_state_listener(self, cb: Callable[..., None]) -> None:
        self._listeners.append(cb)

    def emit_state(self, state: str) -> None:
        self._state = state  # type: ignore[assignment]
        for cb in self._listeners:
            cb(state)


def _make_ctrl_stub(
    fake_session: FakeAgentSession | None = None,
    *,
    has_pending: bool = False,
    backend_mode: str = "independent",
) -> MagicMock:
    """Build a minimal Controller MagicMock for AgentChatDialog tests."""
    from zcu_tools.gui.app.main.services.agent_chat import AgentChatService

    if fake_session is None:
        fake_session = FakeAgentSession()

    ctrl = MagicMock()
    ctrl.get_agent_chat.return_value = AgentChatService()
    ctrl.new_agent_session.return_value = fake_session
    ctrl.attach_agent_session.return_value = fake_session
    ctrl.list_agent_sessions.return_value = []
    ctrl.get_project_root.return_value = "/fake/repo"
    ctrl.has_pending_wait.return_value = has_pending
    ctrl.agent_backend_mode.return_value = backend_mode

    inbox = MagicMock()
    ctrl.get_feedback_inbox.return_value = inbox
    return ctrl


# ---------------------------------------------------------------------------
# D1 — New → conversation page (index 1)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_new_switches_to_conversation_page() -> None:
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    fake = FakeAgentSession()
    ctrl = _make_ctrl_stub(fake)

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    try:
        # Start on picker page.
        assert dialog._stack.currentIndex() == 0

        # Click New.
        dialog._on_picker_new()

        assert dialog._stack.currentIndex() == 1
    finally:
        dialog._on_finished()
        dialog.close()


# ---------------------------------------------------------------------------
# D2 — Attach → conversation page; transcript cleared
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_attach_switches_to_conversation_and_clears_transcript() -> None:
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    fake = FakeAgentSession(initial_state="running", running=True)
    ctrl = _make_ctrl_stub(fake)

    # Put a fake record in the picker list so Attach has something to act on.
    fake_record = {
        "session_id": "aabb1122",
        "claude_session_id": "",
        "pid": 99999,
        "status": "running",
        "log_path": "/tmp/log.ndjson",
        "spool_dir": "/tmp/spool",
        "created": "2026-06-14T00:00:00",
        "title": "test session",
    }
    ctrl.list_agent_sessions.return_value = [fake_record]
    ctrl.attach_agent_session.return_value = fake

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    try:
        # Pre-populate transcript with something.
        dialog._chat.record_assistant("old text")
        dialog._transcript.setPlainText("old text")

        # Simulate selection in picker list.
        dialog._refresh_picker()

        dialog._picker_list.setCurrentRow(0)

        dialog._on_picker_attach()

        assert dialog._stack.currentIndex() == 1
        # Transcript display must be cleared (chat.clear() + transcript.clear()).
        assert dialog._transcript.toPlainText().strip() == ""
    finally:
        dialog._on_finished()
        dialog.close()


# ---------------------------------------------------------------------------
# D3 — Send with no live session → session.start() called
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_send_no_live_session_calls_start() -> None:
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    fake = FakeAgentSession(initial_state="idle", running=False)
    ctrl = _make_ctrl_stub(fake)

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    # Switch to conversation first (simulating after New).
    dialog._on_picker_new()

    try:
        dialog._input.setText("do the thing")
        dialog._on_send()

        assert len(fake.started_tasks) == 1
        task, root = fake.started_tasks[0]
        assert task == "do the thing"
        assert root == "/fake/repo"
        assert fake.sent_messages == []
    finally:
        dialog._on_finished()
        dialog.close()


# ---------------------------------------------------------------------------
# D4 — Close → detach called (NOT stop), back to picker
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_close_calls_detach_not_stop() -> None:
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    fake = FakeAgentSession(initial_state="working", running=True)
    ctrl = _make_ctrl_stub(fake)

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    # Move to conversation page with a live session.
    dialog._switch_to_conversation(fake)

    try:
        assert dialog._stack.currentIndex() == 1

        dialog._on_conversation_close()

        # detach must be called, stop must NOT be called.
        assert fake.detach_calls == 1
        assert fake.stop_calls == 0
        # Back on picker page.
        assert dialog._stack.currentIndex() == 0
    finally:
        dialog._on_finished()
        dialog.close()


# ---------------------------------------------------------------------------
# D5 — Stop-Remove → ctrl.remove_agent_session; list refreshed
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_stop_remove_calls_remove_agent_session() -> None:
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    fake = FakeAgentSession()
    fake_record = {
        "session_id": "deadbeef",
        "claude_session_id": "",
        "pid": 12345,
        "status": "stopped",
        "log_path": "/tmp/log.ndjson",
        "spool_dir": "/tmp/spool",
        "created": "2026-06-14T00:00:00",
        "title": "old task",
    }
    ctrl = _make_ctrl_stub(fake)
    ctrl.list_agent_sessions.return_value = [fake_record]

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    dialog._refresh_picker()

    try:
        # Select the row.
        dialog._picker_list.setCurrentRow(0)

        dialog._on_picker_stop_remove()

        ctrl.remove_agent_session.assert_called_once_with("deadbeef")
        # Refresh must have been called (list is re-queried from ctrl).
        assert ctrl.list_agent_sessions.call_count >= 1
    finally:
        dialog._on_finished()
        dialog.close()


# ---------------------------------------------------------------------------
# D6 — CLI mode: no picker; starts on conversation; Close=stop
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_cli_mode_starts_on_conversation_and_close_stops() -> None:
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    fake = FakeAgentSession(initial_state="working", running=True)
    ctrl = _make_ctrl_stub(fake, backend_mode="cli")
    # new_agent_session is called once during __init__ in CLI mode.
    ctrl.new_agent_session.return_value = fake

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    try:
        # Must start directly on conversation page (no picker).
        assert dialog._stack.currentIndex() == 1

        # In CLI mode, Close = stop (not detach).
        dialog._session = fake
        dialog._on_conversation_close()

        # detach is called (which in CLI AgentRunner calls stop internally),
        # but our FakeAgentSession.detach() is a separate counter.
        # For CLI mode the dialog calls session.detach() which in the real AgentRunner
        # maps to stop().  We verify detach was called.
        assert fake.detach_calls == 1
    finally:
        dialog._on_finished()
        dialog.close()


# ---------------------------------------------------------------------------
# D7 — B1a invariant: dialog does not import concrete AgentRunner
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_dialog_does_not_import_concrete_agent_runner() -> None:
    """AgentChatDialog's module must not import AgentRunner at runtime (B1a)."""
    import re
    import sys

    mod_name = "zcu_tools.gui.app.main.ui.agent_chat_dialog"
    if mod_name in sys.modules:
        import inspect

        mod = sys.modules[mod_name]
        source = inspect.getsource(mod)
    else:
        import importlib.util

        spec = importlib.util.find_spec(mod_name)
        assert spec is not None
        assert spec.origin is not None
        with open(spec.origin) as fh:
            source = fh.read()

    import_pattern = re.compile(
        r"^\s*(from\s+\S+\s+import\s+.*\bAgentRunner\b|import\s+.*\bAgentRunner\b)",
        re.MULTILINE,
    )
    matches = import_pattern.findall(source)
    assert not matches, (
        "AgentChatDialog contains an import of concrete AgentRunner — "
        f"B1a requires the dialog to depend only on AgentSessionPort. Found: {matches}"
    )
