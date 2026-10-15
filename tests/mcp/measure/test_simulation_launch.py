"""A simulation path configures a new process, never an attached GUI."""

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


def orientation(method, params):
    if method.startswith("state."):
        return {"value": False}
    return {"tab.snapshot": {"tabs": []}, "context.active": {"label": None}, "device.list": {"devices": []}}[method]


@pytest.mark.parametrize("launch", ["never", "if_missing"])
def test_connected_attach_rejects_path_without_altering_gui(tmp_path, launch):
    client = make_client(tmp_path, orientation)
    try:
        client.context.session.ensure_connected()
        before = list(client.transport.sent)
        with pytest.raises((ValueError, GuiRpcError), match='unchanged.*launch="new"'):
            client.call("connect", {"launch": launch, "simulate": "/tmp/env.py"})
        assert client.transport.sent == before
        assert client.context.bridge.is_connected
    finally:
        client.context.session.close()


@pytest.mark.parametrize("launch", ["never", "if_missing"])
def test_existing_gui_attach_rejects_path_before_connect(tmp_path, launch, monkeypatch):
    client = make_client(tmp_path, orientation, port_is_open=lambda port: True)
    client.context.bridge.disconnect()
    monkeypatch.setattr(client.context.bridge, "connect", lambda port, token=None: client.context.bridge.set_transport(client.transport))
    try:
        with pytest.raises((ValueError, GuiRpcError), match='unchanged.*launch="new"'):
            client.call("connect", {"launch": launch, "simulate": "/tmp/env.py"})
        assert not client.context.bridge.is_connected
    finally:
        client.context.session.close()


@pytest.mark.parametrize("launch", ["new", "if_missing"])
def test_new_process_receives_simulate_path_only_on_launch(tmp_path, launch, monkeypatch):
    client = make_client(tmp_path, orientation)
    client.context.bridge.disconnect()
    received = []

    def launch_gui(root, port, *, token=None, auto_connect=True, extra_args=None):
        received.append(extra_args)
        client.context.bridge.set_transport(client.transport)

    monkeypatch.setattr(client.context.bridge, "launch", launch_gui)
    try:
        result = client.call("connect", {"launch": launch, "simulate": "/tmp/env.py", "clean": True})
        assert result["launched"]
        assert received == [["--clean", "--simulate", "/tmp/env.py"]]
    finally:
        client.context.session.close()
