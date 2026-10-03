"""Onetone promises through the shipped recipe tools and recording GUI."""

import pytest
from zcu_tools.mcp.core.reply import ToolReply

from ._recipe_support import LookbackGui, _scalar, _section
from ._support import make_client


class OnetoneGui(LookbackGui):
    def __init__(self):
        super().__init__()
        self.publication["tree"]["children"]["sweep"] = _section(
            freq=_section(start=_scalar(4500.0), stop=_scalar(5500.0), expts=_scalar(41))
        )

    def __call__(self, method, params):
        if method == "tab.new":
            assert params == {"adapter_name": "onetone/freq"}
            return {"tab_id": "t"}
        reply = super().__call__(method, params)
        if method == "tab.snapshot":
            reply["tabs"][0]["adapter_name"] = "onetone/freq"
        return reply


@pytest.mark.parametrize("reuse_tab_id", [None, "t"])
def test_onetone_reports_missing_frequency_without_running(tmp_path, reuse_tab_id):
    gui = OnetoneGui()
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("onetone_spectrum", {"reuse_tab_id": reuse_tab_id})
        assert isinstance(reply, ToolReply)
        assert reply.data["status"] == "needs_parameters"
        assert {item["parameter"] for item in reply.data["missing"]} == {
            "center_mhz", "span_mhz"
        }
        assert not reply.is_error
        assert not gui.ran
        methods = [method for method, _ in client.transport.sent]
        assert ("tab.reset_cfg" in methods) is (reuse_tab_id is not None)
        assert ("tab.new" in methods) is (reuse_tab_id is None)
    finally:
        client.context.session.close()
