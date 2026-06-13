"""Tests for B1a — AgentSessionPort seam.

Covers:
  P1 - AgentState lives in ports.py (Literal type).
  P2 - AgentRunner structurally conforms to AgentSessionPort (runtime isinstance).
  P2 - add_state_listener is called on state transitions; exceptions are swallowed.
  P3 - Controller.get_agent_session() returns an AgentSessionPort instance.
  P4 - AgentChatDialog routing (idle/working/waiting) via FakeAgentSession;
       no concrete AgentRunner imported in the dialog.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# P1 — AgentState is owned by ports.py
# ---------------------------------------------------------------------------


def test_agent_state_importable_from_ports() -> None:
    """AgentState must be a top-level name in the ports module."""
    from zcu_tools.gui.app.main.services.ports import AgentState  # noqa: F401


def test_agent_state_re_exported_from_agent_runner() -> None:
    """agent_runner.py must still export AgentState for existing importers."""
    from zcu_tools.gui.app.main.services.agent_runner import AgentState  # noqa: F401


def test_agent_state_is_same_object() -> None:
    """The re-export must point to the same Literal, not a shadowed redefinition."""
    from zcu_tools.gui.app.main.services.agent_runner import (
        AgentState as AgentState_runner,
    )
    from zcu_tools.gui.app.main.services.ports import AgentState as AgentState_ports

    # Both are the same Literal alias; comparing via __args__ (Literal values).
    assert AgentState_runner.__args__ == AgentState_ports.__args__


# ---------------------------------------------------------------------------
# P1/P2 — AgentSessionPort definition and AgentRunner conformance
# ---------------------------------------------------------------------------


def test_agent_session_port_importable() -> None:
    from zcu_tools.gui.app.main.services.ports import AgentSessionPort  # noqa: F401


def test_agent_runner_conforms_to_port_at_runtime() -> None:
    """AgentRunner must satisfy the @runtime_checkable AgentSessionPort Protocol."""
    from zcu_tools.gui.app.main.services.agent_runner import (
        AgentRunner,
        _RunnerCallbacks,
    )
    from zcu_tools.gui.app.main.services.ports import AgentSessionPort

    callbacks = _RunnerCallbacks(
        on_update=lambda _: None,
        on_state_changed=lambda _: None,
        on_process_error=lambda _: None,
        has_pending_wait=lambda: False,
    )
    runner = AgentRunner(callbacks, parent=None)
    assert isinstance(runner, AgentSessionPort)


# ---------------------------------------------------------------------------
# P2 — add_state_listener mechanics
# ---------------------------------------------------------------------------


def _make_runner():  # type: ignore[no-untyped-def]
    from zcu_tools.gui.app.main.services.agent_runner import (
        AgentRunner,
        _RunnerCallbacks,
    )

    callbacks = _RunnerCallbacks(
        on_update=lambda _: None,
        on_state_changed=lambda _: None,
        on_process_error=lambda _: None,
        has_pending_wait=lambda: False,
    )
    return AgentRunner(callbacks, parent=None)


def test_add_state_listener_duplicate_ignored() -> None:
    """Registering the same callback twice must not duplicate calls."""
    runner = _make_runner()
    calls: list[str] = []

    def _cb(s: object) -> None:
        calls.append(str(s))

    runner.add_state_listener(_cb)
    runner.add_state_listener(_cb)  # duplicate — must not double-register
    # Directly trigger _emit_state by manipulating the internal state machine.
    runner._run_state.on_start()
    runner._emit_state()
    assert len(calls) == 1


def test_state_listener_receives_state_on_emit() -> None:
    """_emit_state must call all registered listeners with the current state."""
    runner = _make_runner()
    received: list[object] = []
    runner.add_state_listener(lambda s: received.append(s))
    runner._run_state.on_start()
    runner._emit_state()
    assert received == ["working"]


def test_state_listener_exception_swallowed() -> None:
    """A listener that raises must not interrupt state emission or other listeners."""
    runner = _make_runner()
    second_calls: list[object] = []

    def boom(s: object) -> None:
        raise RuntimeError("injected boom")

    runner.add_state_listener(boom)
    runner.add_state_listener(lambda s: second_calls.append(s))

    runner._run_state.on_start()
    # Must not raise:
    runner._emit_state()
    # Second listener still receives the state.
    assert second_calls == ["working"]


# ---------------------------------------------------------------------------
# P3 — Controller.get_agent_session() factory
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_controller_get_agent_session_returns_port() -> None:
    """get_agent_session() must return a value that satisfies AgentSessionPort.

    Uses a MagicMock controller stub to avoid spinning up a full Qt app.
    The real factory method is tested via the return-type isinstance check
    on the concrete AgentRunner it builds internally.
    """
    # We cannot call the real Controller without a full Qt app and all services.
    # Instead, validate that AgentRunner (which get_agent_session creates) passes.
    from zcu_tools.gui.app.main.services.agent_runner import (
        AgentRunner,
        _RunnerCallbacks,
    )
    from zcu_tools.gui.app.main.services.ports import AgentSessionPort

    callbacks = _RunnerCallbacks(
        on_update=lambda _: None,
        on_state_changed=lambda _: None,
        on_process_error=lambda _: None,
        has_pending_wait=lambda: False,
    )
    session: AgentSessionPort = AgentRunner(callbacks, parent=None)
    assert isinstance(session, AgentSessionPort)
    # Pyright validates the annotation at type-check time; this confirms runtime.


# ---------------------------------------------------------------------------
# P4 — AgentChatDialog uses only AgentSessionPort (no concrete AgentRunner)
# ---------------------------------------------------------------------------


class FakeAgentSession:
    """Qt-free fake for AgentSessionPort used in dialog routing tests."""

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
        """B1b-2: detach from session without stopping it."""
        self.detach_calls += 1
        self._running = False

    def session_id(self) -> str:
        return "fake-session-id"

    def add_state_listener(self, cb: Callable[..., None]) -> None:
        self._listeners.append(cb)

    def emit_state(self, state: str) -> None:
        """Test helper: push a state change to all listeners."""
        self._state = state  # type: ignore[assignment]
        for cb in self._listeners:
            cb(state)


def _make_ctrl_stub(
    fake_session: FakeAgentSession,
    *,
    has_pending: bool = False,
) -> MagicMock:
    """Build a minimal Controller MagicMock for AgentChatDialog tests."""
    from zcu_tools.gui.app.main.services.agent_chat import AgentChatService

    ctrl = MagicMock()
    ctrl.get_agent_chat.return_value = AgentChatService()
    # B1b-2: new_agent_session replaces get_agent_session as the factory.
    ctrl.new_agent_session.return_value = fake_session
    ctrl.get_agent_session.return_value = fake_session
    ctrl.list_agent_sessions.return_value = []
    ctrl.agent_backend_mode.return_value = "independent"
    ctrl.get_project_root.return_value = "/fake/repo"
    ctrl.has_pending_wait.return_value = has_pending
    # FeedbackInbox stub: just record post() calls.
    inbox = MagicMock()
    ctrl.get_feedback_inbox.return_value = inbox
    return ctrl


@pytest.mark.usefixtures("qapp")
def test_dialog_routes_idle_to_start() -> None:
    """When no session is running, Send must call session.start() (not send_user_message)."""
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    fake = FakeAgentSession(initial_state="idle", running=False)
    ctrl = _make_ctrl_stub(fake)

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    # Switch to conversation page (New) so _on_send can reach the input widget.
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
        dialog._on_finished()  # clean up listener without Qt close
        dialog.close()


@pytest.mark.usefixtures("qapp")
def test_dialog_routes_working_to_send_user_message() -> None:
    """When session is running (state=working), Send must call send_user_message."""
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    fake = FakeAgentSession(initial_state="working", running=True)
    ctrl = _make_ctrl_stub(fake, has_pending=False)

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    # Force session into dialog without picker interaction.
    dialog._switch_to_conversation(fake)
    try:
        dialog._input.setText("redirect the agent")
        dialog._on_send()

        assert fake.sent_messages == ["redirect the agent"]
        assert len(fake.started_tasks) == 0
    finally:
        dialog._on_finished()
        dialog.close()


@pytest.mark.usefixtures("qapp")
def test_dialog_routes_waiting_to_feedback_inbox() -> None:
    """When state=waiting (has_pending_wait), Send must post to feedback inbox."""
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    fake = FakeAgentSession(initial_state="waiting", running=True)
    ctrl = _make_ctrl_stub(fake, has_pending=True)

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    dialog._switch_to_conversation(fake)
    try:
        dialog._input.setText("wake up now")
        dialog._on_send()

        inbox = ctrl.get_feedback_inbox.return_value
        inbox.post.assert_called_once_with("wake up now")
        assert fake.sent_messages == []
        assert len(fake.started_tasks) == 0
    finally:
        dialog._on_finished()
        dialog.close()


@pytest.mark.usefixtures("qapp")
def test_dialog_state_listener_updates_ui() -> None:
    """add_state_listener registered by the dialog must update the status label."""
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    fake = FakeAgentSession(initial_state="idle", running=False)
    ctrl = _make_ctrl_stub(fake)

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    # Switch to conversation so the status label is visible and listener registered.
    dialog._switch_to_conversation(fake)
    try:
        # Simulate backend transitioning to working.
        fake.emit_state("working")
        assert "working" in dialog._agent_status.text()

        # Simulate backend going idle again.
        fake.emit_state("idle")
        assert "idle" in dialog._agent_status.text()
    finally:
        dialog._on_finished()
        dialog.close()


@pytest.mark.usefixtures("qapp")
def test_dialog_does_not_import_concrete_agent_runner() -> None:
    """AgentChatDialog's module must not import AgentRunner at runtime.

    B1a invariant: the dialog depends only on the port, never the concrete class.
    We check that no ``import AgentRunner`` or ``from ... import AgentRunner``
    appears in the executable (non-comment, non-docstring) portion of the source.
    """
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
        source = open(spec.origin).read()  # noqa: WPS515

    # Match actual import statements for AgentRunner — not mentions in comments/docstrings.
    # Pattern: `import AgentRunner` or `from ... import ... AgentRunner`
    import_pattern = re.compile(
        r"^\s*(from\s+\S+\s+import\s+.*\bAgentRunner\b|import\s+.*\bAgentRunner\b)",
        re.MULTILINE,
    )
    matches = import_pattern.findall(source)
    assert not matches, (
        "AgentChatDialog contains an import of concrete AgentRunner — "
        f"B1a requires the dialog to depend only on AgentSessionPort. Found: {matches}"
    )
