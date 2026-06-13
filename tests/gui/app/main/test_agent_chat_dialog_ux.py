"""Tests for Issue 3 (transcript colouring) and Issue 4 (HistoryLineEdit) UX.

Coverage
--------
  H1  - push_history: empty string is not stored.
  H2  - push_history: consecutive duplicate is not stored.
  H3  - push_history: different messages are stored in order.
  H4  - Up navigates to the most-recent entry; repeated Up goes to oldest.
  H5  - Down restores the draft after navigating Up past the oldest entry.
  H6  - Down past newest entry restores draft, not a blank string.
  H7  - Up when history is empty does nothing.
  H8  - Down at the draft position (index == len) does nothing.
  H9  - Draft is saved from the actual field text (not empty string).

  F1  - entry_format returns correct (color, prefix) for every known kind.
  F2  - entry_format returns a fallback tuple for an unknown kind.
  F3  - transcript uses colour / prefix via _append_colored_entry
        (light smoke: document is non-empty after one entry is appended).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# H-series: HistoryLineEdit — all tests use qapp so Qt widgets initialise
# cleanly, then exercise the history logic via the public text()/setText() API.
# ---------------------------------------------------------------------------


@pytest.fixture()
def history_widget(qapp):  # type: ignore[no-untyped-def]
    """Fresh HistoryLineEdit with a real QApp."""
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import HistoryLineEdit

    return HistoryLineEdit()


@pytest.mark.usefixtures("qapp")
def test_H1_push_history_rejects_empty(history_widget) -> None:  # type: ignore[no-untyped-def]
    history_widget.push_history("")
    assert history_widget._history == []


@pytest.mark.usefixtures("qapp")
def test_H2_push_history_rejects_consecutive_duplicate(history_widget) -> None:  # type: ignore[no-untyped-def]
    history_widget.push_history("hello")
    history_widget.push_history("hello")
    assert history_widget._history == ["hello"]


@pytest.mark.usefixtures("qapp")
def test_H3_push_history_stores_different_messages_in_order(history_widget) -> None:  # type: ignore[no-untyped-def]
    history_widget.push_history("first")
    history_widget.push_history("second")
    history_widget.push_history("third")
    assert history_widget._history == ["first", "second", "third"]


@pytest.mark.usefixtures("qapp")
def test_H4_up_navigates_from_newest_to_oldest(history_widget) -> None:  # type: ignore[no-untyped-def]
    w = history_widget
    w.push_history("a")
    w.push_history("b")
    w.push_history("c")

    w._navigate_up()
    assert w.text() == "c"

    w._navigate_up()
    assert w.text() == "b"

    w._navigate_up()
    assert w.text() == "a"

    # Extra Up at oldest — stays at "a".
    w._navigate_up()
    assert w.text() == "a"
    assert w._index == 0


@pytest.mark.usefixtures("qapp")
def test_H5_down_restores_draft_after_up(history_widget) -> None:  # type: ignore[no-untyped-def]
    w = history_widget
    w.push_history("msg1")
    w.push_history("msg2")

    # Simulate the user having typed something before pressing Up.
    w.setText("partial draft")

    w._navigate_up()  # saves draft, shows "msg2"
    assert w._draft == "partial draft"
    assert w.text() == "msg2"

    w._navigate_up()  # shows "msg1"
    assert w.text() == "msg1"

    w._navigate_down()  # back to "msg2"
    assert w.text() == "msg2"

    w._navigate_down()  # past newest → restore draft
    assert w.text() == "partial draft"
    assert w._index == len(w._history)


@pytest.mark.usefixtures("qapp")
def test_H6_draft_is_not_blank_when_user_had_typed(history_widget) -> None:  # type: ignore[no-untyped-def]
    w = history_widget
    w.push_history("sent")

    w.setText("in progress")
    w._navigate_up()  # draft = "in progress"
    w._navigate_down()  # restore draft

    assert w.text() == "in progress"


@pytest.mark.usefixtures("qapp")
def test_H7_up_with_empty_history_does_nothing(history_widget) -> None:  # type: ignore[no-untyped-def]
    w = history_widget
    w.setText("something")
    w._navigate_up()
    # No change — history is empty; index stays 0.
    assert w._index == 0
    assert w.text() == "something"


@pytest.mark.usefixtures("qapp")
def test_H8_down_at_draft_position_does_nothing(history_widget) -> None:  # type: ignore[no-untyped-def]
    w = history_widget
    w.push_history("only")
    # Start already at draft position (index == len == 1).
    assert w._index == len(w._history)

    w.setText("draft")
    w._navigate_down()  # index is already at len; should not change.
    assert w._index == len(w._history)
    assert w.text() == "draft"


@pytest.mark.usefixtures("qapp")
def test_H9_draft_is_saved_from_actual_field_text(history_widget) -> None:  # type: ignore[no-untyped-def]
    w = history_widget
    w.push_history("x")

    w.setText("typed before up")
    w._navigate_up()

    assert w._draft == "typed before up"


# ---------------------------------------------------------------------------
# F-series: entry_format (pure function, no QApp)
# ---------------------------------------------------------------------------


def test_F1_entry_format_returns_known_kinds() -> None:
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import entry_format

    known_kinds = [
        "feedback",
        "assistant",
        "tool_use",
        "tool_result",
        "system",
        "result",
        "activity",
        "diagnostic",
    ]
    for kind in known_kinds:
        color, prefix = entry_format(kind)
        assert color.startswith("#"), (
            f"kind={kind!r}: expected hex color, got {color!r}"
        )
        assert isinstance(prefix, str), f"kind={kind!r}: prefix must be str"


def test_F1_feedback_and_assistant_differ() -> None:
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import entry_format

    fb_color, _ = entry_format("feedback")
    ast_color, _ = entry_format("assistant")
    # User and agent must have visually distinct colours.
    assert fb_color != ast_color


def test_F1_diagnostic_differs_from_secondary() -> None:
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import entry_format

    diag_color, _ = entry_format("diagnostic")
    tool_color, _ = entry_format("tool_use")
    # Warning colour must stand out from secondary grey.
    assert diag_color != tool_color


def test_F2_entry_format_fallback_for_unknown_kind() -> None:
    from zcu_tools.gui.app.main.ui.agent_chat_dialog import entry_format

    color, prefix = entry_format("some_future_kind_xyz")
    assert color.startswith("#")
    assert isinstance(prefix, str)


# ---------------------------------------------------------------------------
# F3: smoke — _append_colored_entry populates transcript document (needs QApp)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_F3_append_colored_entry_populates_document() -> None:
    """Smoke-test that _append_colored_entry writes content into the QPlainTextEdit."""
    from unittest.mock import MagicMock

    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    ctrl = MagicMock()
    from zcu_tools.gui.app.main.services.agent_chat import AgentChatService

    ctrl.get_agent_chat.return_value = AgentChatService()
    ctrl.new_agent_session.return_value = MagicMock(
        state="idle",
        is_running=lambda: False,
    )
    ctrl.list_agent_sessions.return_value = []
    ctrl.agent_backend_mode.return_value = "independent"

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    try:
        # Document starts empty.
        doc = dialog._transcript.document()
        assert doc is not None and doc.isEmpty()

        # Record an assistant entry into the service.
        dialog._chat.record_assistant("hello from agent")
        # Force refresh so the widget picks it up.
        dialog._refresh_all()

        text = dialog._transcript.toPlainText()
        assert "hello from agent" in text
    finally:
        dialog._on_finished()
        dialog.close()


@pytest.mark.usefixtures("qapp")
def test_F3_multiple_entries_all_appear_in_transcript() -> None:
    """All kinds of entries must appear in the transcript text after _refresh_all."""
    from unittest.mock import MagicMock

    from zcu_tools.gui.app.main.ui.agent_chat_dialog import AgentChatDialog

    ctrl = MagicMock()
    from zcu_tools.gui.app.main.services.agent_chat import AgentChatService

    svc = AgentChatService()
    ctrl.get_agent_chat.return_value = svc
    ctrl.new_agent_session.return_value = MagicMock(
        state="idle",
        is_running=lambda: False,
    )
    ctrl.list_agent_sessions.return_value = []
    ctrl.agent_backend_mode.return_value = "independent"

    dialog = AgentChatDialog(ctrl)  # type: ignore[arg-type]
    try:
        svc.record_assistant("agent reply")
        svc.record_feedback("user message")
        # record_diagnostic writes when _embedded_active is False (default).
        svc.record_diagnostic("error", "title", "something went wrong")

        dialog._refresh_all()

        text = dialog._transcript.toPlainText()
        assert "agent reply" in text
        assert "user message" in text
        assert "something went wrong" in text
    finally:
        dialog._on_finished()
        dialog.close()
