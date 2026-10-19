"""Startup and data-root options configure new processes, never attached GUIs."""

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


def orientation(method, params):
    if method.startswith("state."):
        return {"value": False}
    return {
        "tab.snapshot": {"tabs": []},
        "context.active": {"label": None},
        "device.list": {"devices": []},
        "operation.active": {"operations": []},
    }[method]


@pytest.mark.parametrize("launch", ["never", "if_missing"])
@pytest.mark.parametrize("option", ["script", "project_root"])
def test_connected_attach_rejects_path_without_altering_gui(tmp_path, launch, option):
    client = make_client(tmp_path, orientation)
    try:
        client.context.session.ensure_connected()
        before = list(client.transport.sent)
        with pytest.raises((ValueError, GuiRpcError), match='unchanged.*launch="new"'):
            client.call("connect", {"launch": launch, option: "/tmp/isolated"})
        assert client.transport.sent == before
        assert client.context.bridge.is_connected
    finally:
        client.context.session.close()


@pytest.mark.parametrize("launch", ["never", "if_missing"])
@pytest.mark.parametrize("option", ["script", "project_root"])
def test_existing_gui_attach_rejects_path_before_connect(
    tmp_path, launch, option, monkeypatch
):
    client = make_client(tmp_path, orientation, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    monkeypatch.setattr(
        client.context.bridge,
        "connect",
        lambda port, token=None: client.context.bridge.set_transport(client.transport),
    )
    try:
        with pytest.raises((ValueError, GuiRpcError), match='unchanged.*launch="new"'):
            client.call("connect", {"launch": launch, option: "/tmp/isolated"})
        assert not client.context.bridge.is_connected
    finally:
        client.context.session.close()


@pytest.mark.parametrize("launch", ["new", "if_missing"])
def test_new_process_receives_script_path_only_on_launch(tmp_path, launch, monkeypatch):
    client = make_client(tmp_path, orientation)
    client.context.bridge.set_transport(None)
    received = []

    def launch_gui(root, port, *, token=None, auto_connect=True, extra_args=None):
        received.append(extra_args)
        client.context.bridge.set_transport(client.transport)

    monkeypatch.setattr(client.context.bridge, "launch", launch_gui)
    try:
        result = client.call("connect", {"launch": launch, "script": "/tmp/env.py"})
        assert result["launched"]
        assert received == [["--script", "/tmp/env.py"]]
    finally:
        client.context.session.close()


@pytest.mark.parametrize("launch", ["new", "if_missing"])
@pytest.mark.parametrize("project_root", ["/isolated project", "relative", "", "\0"])
def test_new_process_receives_project_root_without_mcp_path_validation(
    tmp_path, launch, project_root, monkeypatch
):
    client = make_client(tmp_path, orientation)
    client.context.bridge.set_transport(None)
    received = []

    def launch_gui(root, port, *, token=None, auto_connect=True, extra_args=None):
        received.append((root, extra_args))
        client.context.bridge.set_transport(client.transport)

    monkeypatch.setattr(client.context.bridge, "launch", launch_gui)
    try:
        result = client.call(
            "connect",
            {
                "launch": launch,
                "project_root": project_root,
                "script": "relative/startup.py",
            },
        )
        assert result["launched"]
        assert len(received) == 1
        code_root, arguments = received[0]
        assert str(code_root) != project_root
        assert arguments == [
            "--script",
            "relative/startup.py",
            "--project-root",
            project_root,
        ]
    finally:
        client.context.session.close()


@pytest.mark.parametrize("project_root", [False, 12, [], {}])
def test_invalid_project_root_type_is_rejected_before_rpc(tmp_path, project_root):
    client = make_client(tmp_path, orientation)
    try:
        client.context.session.ensure_connected()
        before = list(client.transport.sent)
        with pytest.raises(ValueError, match="project_root.*string"):
            client.call("connect", {"project_root": project_root})
        assert client.transport.sent == before
    finally:
        client.context.session.close()


@pytest.mark.parametrize("clean, expected", [(False, []), (True, ["--clean"])])
def test_launch_without_script_delegates_default_or_clean_selection(
    tmp_path, monkeypatch, clean, expected
):
    client = make_client(tmp_path, orientation)
    client.context.bridge.set_transport(None)
    received = []

    def launch_gui(root, port, *, token=None, auto_connect=True, extra_args=None):
        received.append(extra_args)
        client.context.bridge.set_transport(client.transport)

    monkeypatch.setattr(client.context.bridge, "launch", launch_gui)
    try:
        result = client.call("connect", {"launch": "new", "clean": clean})
        assert result["launched"]
        assert received == [expected]
        client.call("connect", {})
        assert received == [expected]
    finally:
        client.context.session.close()


def test_clean_with_script_rejected_before_network(tmp_path):
    client = make_client(tmp_path, orientation)
    try:
        with pytest.raises(ValueError, match="clean.*script"):
            client.call(
                "connect", {"launch": "new", "clean": True, "script": "start.py"}
            )
        assert client.transport.sent == []
    finally:
        client.context.session.close()


@pytest.mark.parametrize("script", ["", False, 1, []])
def test_invalid_script_rejected_before_network(tmp_path, script):
    client = make_client(tmp_path, orientation)
    try:
        with pytest.raises(ValueError, match="script.*non-empty"):
            client.call("connect", {"launch": "new", "script": script})
        assert client.transport.sent == []
    finally:
        client.context.session.close()
