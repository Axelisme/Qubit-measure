"""MetaDict tools preserve GUI state, bounded summaries and confirmed prefix."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.mark.uses_wall_clock
def test_md_tools_share_gui_values_and_stop_after_a_failed_key(
    qapp, tmp_path: Path
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke("project", {"chip": "chip", "qubit": "q", "resonator": "res"})
        invoke("context_create", {"label": "base"})
        assert invoke("md_get", {}) == {"values": {}}

        matrix = [[1, 2], [3, 4]]
        assert invoke("md_set", {"values": {"freq": 5.0, "matrix": matrix}}) == {
            "freq": {"before": None, "after": 5.0},
            "matrix": {"before": None, "after": matrix},
        }
        assert invoke("md_get", {}) == {
            "values": {"freq": 5.0, "matrix": "2 × 2 matrix"}
        }
        assert invoke("md_get", {"keys": ["matrix"]}) == {"values": {"matrix": matrix}}
        assert (
            call(sock, "context.md_get_attr", {"key": "freq"})["result"]["value"] == 5.0
        )
        with pytest.raises(GuiRpcError, match="missing"):
            invoke("md_get", {"keys": ["missing"]})

        with pytest.raises(GuiRpcError, match="items.*freq.*before.*after"):
            invoke("md_set", {"values": {"freq": 6.0, "items": 7, "later": 8}})
        assert (
            call(sock, "context.md_get_attr", {"key": "freq"})["result"]["value"] == 6.0
        )
        assert call(sock, "context.md_get", {})["result"]["keys"] == ["freq", "matrix"]
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()
