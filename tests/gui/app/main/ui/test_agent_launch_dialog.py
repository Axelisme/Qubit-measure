"""Tests for ``AgentLaunchDialog`` — the two-button external-terminal launcher.

Headless (offscreen qapp from tests/gui/conftest.py). The launcher module is
monkeypatched so no terminal is spawned; the tests assert the Resume button's
enabled state and that each button calls ``launch_agent_terminal`` with the
right ``resume`` flag and the controller's project root.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services import agent_launcher
from zcu_tools.gui.app.main.ui.agent_launch_dialog import AgentLaunchDialog


def _make_ctrl(project_root: str = "/repo/root") -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_project_root.return_value = project_root
    return ctrl


@pytest.mark.usefixtures("qapp")
def test_resume_disabled_when_no_last_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_launcher, "read_last_session_id", lambda: None)
    dialog = AgentLaunchDialog(_make_ctrl())  # type: ignore[arg-type]
    assert not dialog._resume_btn.isEnabled()
    assert dialog._new_btn.isEnabled()


@pytest.mark.usefixtures("qapp")
def test_resume_enabled_when_last_session_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_launcher, "read_last_session_id", lambda: "sess-1")
    dialog = AgentLaunchDialog(_make_ctrl())  # type: ignore[arg-type]
    assert dialog._resume_btn.isEnabled()


@pytest.mark.usefixtures("qapp")
def test_new_button_launches_with_resume_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_launcher, "read_last_session_id", lambda: None)
    launch = MagicMock(return_value="new-sess")
    monkeypatch.setattr(agent_launcher, "launch_agent_terminal", launch)

    ctrl = _make_ctrl()
    dialog = AgentLaunchDialog(ctrl)  # type: ignore[arg-type]
    dialog._new_btn.click()

    launch.assert_called_once_with("/repo/root", resume=False)
    assert "new-sess" in dialog._status_label.text()
    # A launch persists a "last" session, so Resume becomes available.
    assert dialog._resume_btn.isEnabled()


@pytest.mark.usefixtures("qapp")
def test_resume_button_launches_with_resume_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_launcher, "read_last_session_id", lambda: "sess-1")
    launch = MagicMock(return_value="sess-1")
    monkeypatch.setattr(agent_launcher, "launch_agent_terminal", launch)

    dialog = AgentLaunchDialog(_make_ctrl())  # type: ignore[arg-type]
    dialog._resume_btn.click()

    launch.assert_called_once_with("/repo/root", resume=True)


@pytest.mark.usefixtures("qapp")
def test_launch_failure_shown_in_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_launcher, "read_last_session_id", lambda: None)

    def _boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("no terminal found")

    monkeypatch.setattr(agent_launcher, "launch_agent_terminal", _boom)

    dialog = AgentLaunchDialog(_make_ctrl())  # type: ignore[arg-type]
    dialog._new_btn.click()

    assert "Failed to launch terminal" in dialog._status_label.text()
    assert "no terminal found" in dialog._status_label.text()
