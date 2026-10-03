"""Coherence behavior through shipped tools and the GUI wire boundary."""

from typing import Any

from ._recipe_support import LookbackGui, scalar
from ._support import make_client


class CoherenceGui(LookbackGui):
    def __init__(self):
        super().__init__()
        self.md = {"r_f": 5100.0}
        self.publication["tree"]["children"]["modules"]["children"]["pi_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": "<Custom:Pulse>",
            "error": None,
            "children": {"freq": scalar(6100.0)},
        }

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.new":
            assert params == {"adapter_name": "twotone/t1"}
            return {"tab_id": "t"}
        result = super().__call__(method, params)
        if method == "tab.snapshot":
            result["tabs"][0]["adapter_name"] = "twotone/t1"
        return result


def test_t1_requires_calibrated_pi_instead_of_custom_template(tmp_path):
    gui = CoherenceGui()
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("t1", {})
        assert reply.data["status"] == "needs_parameters", reply.data
        assert {item["parameter"] for item in reply.data["missing"]} == {"pi_ref"}
        assert not gui.ran
    finally:
        client.context.session.close()
