"""Tests for ``AgentLaunchDialog`` — the session-list external-terminal launcher.

Headless (offscreen qapp from tests/gui/conftest.py). The launcher module is
monkeypatched so no terminal is spawned; the tests assert list population,
Resume-selected / New button behaviour, and enabled-state of the Resume button.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import QDialog  # type: ignore[attr-defined]
from zcu_tools.gui.app.main.services import agent_launcher
from zcu_tools.gui.app.main.services.agent_launcher import ResumableSession
from zcu_tools.gui.app.main.ui.agent_launch_dialog import AgentLaunchDialog

_SESSION_ID_ROLE = Qt.ItemDataRole.UserRole

_BOOTSTRAP_PROMPT = "Read live GUI state with gui_overview before doing anything."

_FAKE_SESSIONS = [
    ResumableSession(
        session_id="sess-aaa",
        last_active=2000.0,
        label="First user message of session aaa",
    ),
    ResumableSession(
        session_id="sess-bbb",
        last_active=1000.0,
        label="Earlier session bbb",
    ),
]


def _make_ctrl(project_root: str = "/repo/root") -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_project_root.return_value = project_root
    ctrl.build_agent_bootstrap_prompt.return_value = _BOOTSTRAP_PROMPT
    return ctrl


# ---------------------------------------------------------------------------
# List population
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_dialog_populates_list_from_resumable_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_launcher, "list_resumable_sessions", lambda _root: _FAKE_SESSIONS
    )
    dialog = AgentLaunchDialog(_make_ctrl())  # type: ignore[arg-type]
    assert dialog._session_list.count() == 2
    # First item in the list corresponds to the first (most recent) session.
    item0 = dialog._session_list.item(0)
    assert item0 is not None
    assert "sess-aaa" in item0.data(_SESSION_ID_ROLE)


@pytest.mark.usefixtures("qapp")
def test_dialog_selects_first_item_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_launcher, "list_resumable_sessions", lambda _root: _FAKE_SESSIONS
    )
    dialog = AgentLaunchDialog(_make_ctrl())  # type: ignore[arg-type]
    assert dialog._session_list.currentRow() == 0


@pytest.mark.usefixtures("qapp")
def test_dialog_empty_list_when_no_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_launcher, "list_resumable_sessions", lambda _root: [])
    dialog = AgentLaunchDialog(_make_ctrl())  # type: ignore[arg-type]
    assert dialog._session_list.count() == 0


# ---------------------------------------------------------------------------
# Resume button enabled/disabled state
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_resume_enabled_when_session_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_launcher, "list_resumable_sessions", lambda _root: _FAKE_SESSIONS
    )
    dialog = AgentLaunchDialog(_make_ctrl())  # type: ignore[arg-type]
    # Default selection → Resume enabled.
    assert dialog._resume_btn.isEnabled()


@pytest.mark.usefixtures("qapp")
def test_resume_disabled_when_no_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_launcher, "list_resumable_sessions", lambda _root: [])
    dialog = AgentLaunchDialog(_make_ctrl())  # type: ignore[arg-type]
    assert not dialog._resume_btn.isEnabled()
    # New button is always enabled.
    assert dialog._new_btn.isEnabled()


# ---------------------------------------------------------------------------
# Resume selected button
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_resume_button_launches_with_selected_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_launcher, "list_resumable_sessions", lambda _root: _FAKE_SESSIONS
    )
    launch = MagicMock(return_value="sess-aaa")
    monkeypatch.setattr(agent_launcher, "launch_agent_terminal", launch)

    ctrl = _make_ctrl()
    dialog = AgentLaunchDialog(ctrl)  # type: ignore[arg-type]
    # First item is selected by default; click Resume.
    dialog._resume_btn.click()

    launch.assert_called_once_with(
        "/repo/root",
        resume_session_id="sess-aaa",
        bootstrap_prompt=_BOOTSTRAP_PROMPT,
    )
    assert "sess-aaa" in dialog._status_label.text()


@pytest.mark.usefixtures("qapp")
def test_resume_button_uses_explicitly_selected_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_launcher, "list_resumable_sessions", lambda _root: _FAKE_SESSIONS
    )
    launch = MagicMock(return_value="sess-bbb")
    monkeypatch.setattr(agent_launcher, "launch_agent_terminal", launch)

    ctrl = _make_ctrl()
    dialog = AgentLaunchDialog(ctrl)  # type: ignore[arg-type]
    # Select the second row explicitly.
    dialog._session_list.setCurrentRow(1)
    dialog._resume_btn.click()

    launch.assert_called_once_with(
        "/repo/root",
        resume_session_id="sess-bbb",
        bootstrap_prompt=_BOOTSTRAP_PROMPT,
    )


# ---------------------------------------------------------------------------
# New session button
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_new_button_launches_without_resume_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_launcher, "list_resumable_sessions", lambda _root: [])
    launch = MagicMock(return_value="new-sess")
    monkeypatch.setattr(agent_launcher, "launch_agent_terminal", launch)

    ctrl = _make_ctrl()
    dialog = AgentLaunchDialog(ctrl)  # type: ignore[arg-type]
    dialog._new_btn.click()

    launch.assert_called_once_with(
        "/repo/root",
        resume_session_id=None,
        bootstrap_prompt=_BOOTSTRAP_PROMPT,
    )
    assert "new-sess" in dialog._status_label.text()
    # Launch closes the dialog (accepted).
    assert dialog.result() == QDialog.DialogCode.Accepted


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_launch_failure_shown_in_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_launcher, "list_resumable_sessions", lambda _root: [])

    def _boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("no terminal found")

    monkeypatch.setattr(agent_launcher, "launch_agent_terminal", _boom)

    dialog = AgentLaunchDialog(_make_ctrl())  # type: ignore[arg-type]
    dialog._new_btn.click()

    assert "Failed to launch terminal" in dialog._status_label.text()
    assert "no terminal found" in dialog._status_label.text()
    # Failure keeps the dialog open (not accepted).
    assert dialog.result() != QDialog.DialogCode.Accepted
