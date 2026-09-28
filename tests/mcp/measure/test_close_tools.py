"""Close tools delegate admission and never force the responding GUI to exit."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from zcu_tools.mcp.core import bridge as bridge_module
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


@pytest.mark.parametrize("stopped", [False, True])
def test_shutdown_waits_for_reply_process_without_stop_or_prechecks(
    tmp_path, monkeypatch, stopped
):
    def respond(method, params):
        assert method == "app.shutdown"
        assert params == {"discard_unsaved": True}
        return {"shutting_down": True, "pid": 1234}

    client = make_client(tmp_path, respond)
    wait = Mock(return_value=stopped)
    stop = Mock(side_effect=AssertionError("must not terminate GUI"))
    monkeypatch.setattr(client.context.bridge, "wait_for_gui_exit", wait)
    monkeypatch.setattr(client.context.bridge, "stop", stop)
    assert client.call("shutdown", {"discard_unsaved": True}) == {"stopped": stopped}
    wait.assert_called_once_with(1234, timeout=5.0)
    stop.assert_not_called()
    commands = [method for method, _ in client.transport.sent]
    assert commands.count("app.shutdown") == 1
    assert "operation.active" not in commands
    assert "tab.snapshot" not in commands
    if not stopped:
        assert client.transport.is_open


@pytest.mark.parametrize("reason", ["busy", "unsaved"])
def test_shutdown_admission_failure_never_waits_or_forces(
    tmp_path, monkeypatch, reason
):
    client = make_client(tmp_path)
    client.transport.replies["app.shutdown"] = {
        "ok": False,
        "error": {"code": "precondition_failed", "reason": reason, "message": reason},
    }
    wait = Mock(side_effect=AssertionError("admission failed"))
    monkeypatch.setattr(client.context.bridge, "wait_for_gui_exit", wait)
    with pytest.raises(GuiRpcError) as error:
        client.call("shutdown", {})
    assert error.value.reason == reason
    wait.assert_not_called()
    assert [
        (name, args) for name, args in client.transport.sent if name == "app.shutdown"
    ] == [("app.shutdown", {"discard_unsaved": False})]


def test_shutdown_reply_timeout_returns_unconfirmed_without_retry(
    tmp_path, monkeypatch
):
    client = make_client(tmp_path)
    client.transport.replies["app.shutdown"] = {
        "ok": False,
        "error": {"code": "timeout", "message": "owner did not reply"},
    }
    wait = Mock(side_effect=AssertionError("no confirmed process identity"))
    monkeypatch.setattr(client.context.bridge, "wait_for_gui_exit", wait)
    assert client.call("shutdown", {}) == {"stopped": False}
    wait.assert_not_called()
    assert [name for name, _ in client.transport.sent].count("app.shutdown") == 1


def test_shutdown_transport_timeout_does_not_stop_or_retry(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    client.context.session.ensure_connected()
    sent = []

    def drop_reply(payload):
        sent.append(payload)

    monkeypatch.setattr(client.transport, "send_line", drop_reply)
    clock = iter([0.0, 100.0])
    monkeypatch.setattr(
        bridge_module, "time", SimpleNamespace(monotonic=lambda: next(clock))
    )
    stop = Mock(side_effect=AssertionError("must not force shutdown"))
    wait = Mock(side_effect=AssertionError("no confirmed process identity"))
    monkeypatch.setattr(client.context.bridge, "stop", stop)
    monkeypatch.setattr(client.context.bridge, "wait_for_gui_exit", wait)

    assert client.call("shutdown", {}) == {"stopped": False}

    assert [payload["method"] for payload in sent] == ["app.shutdown"]
    assert not client.transport.is_open
    stop.assert_not_called()
    wait.assert_not_called()


def test_tab_close_is_one_explicit_command(tmp_path):
    def respond(method, params):
        assert method == "tab.close"
        assert params == {"tab_id": "t1", "discard_unsaved": False}
        return {"ok": True}

    client = make_client(tmp_path, respond)
    assert client.call("tab_close", {"tab": "t1"}) == {"closed": "t1"}
    methods = [name for name, _ in client.transport.sent]
    assert methods.count("tab.close") == 1
    assert "tab.snapshot" not in methods
