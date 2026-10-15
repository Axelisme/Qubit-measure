"""A simulation path configures a new process, never an attached GUI."""

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


@pytest.mark.parametrize("launch", ["never", "if_missing"])
def test_connected_attach_rejects_path_without_altering_gui(tmp_path, launch):
    client = make_client(tmp_path)
    try:
        client.call("rpc_call", {"method": "control.status"})
        before = list(client.transport.sent)
        with pytest.raises((ValueError, GuiRpcError), match='unchanged.*launch="new"'):
            client.call("connect", {"launch": launch, "simulate": "/tmp/env.py"})
        assert client.transport.sent == before
        assert client.context.bridge.is_connected
    finally:
        client.context.session.close()


@pytest.mark.parametrize("launch", ["never", "if_missing"])
def test_existing_gui_attach_rejects_path_before_connect(tmp_path, launch):
    client = make_client(tmp_path, port_is_open=lambda port: True)
    client.context.bridge.disconnect()
    try:
        with pytest.raises((ValueError, GuiRpcError), match='unchanged.*launch="new"'):
            client.call("connect", {"launch": launch, "simulate": "/tmp/env.py"})
        assert not client.context.bridge.is_connected
    finally:
        client.context.session.close()
