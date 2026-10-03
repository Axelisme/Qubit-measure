"""GE calibration behavior through shipped tools and the GUI wire boundary."""

from ._recipe_support import LookbackGui, scalar
from ._support import make_client


class GeGui(LookbackGui):
    def __init__(self):
        super().__init__()
        self.md = {"r_f": 5100.0}
        modules = self.publication["tree"]["children"]["modules"]["children"]
        modules["probe_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": "<Custom:Pulse>",
            "error": None,
            "children": {"freq": scalar(6100.0)},
        }

    def __call__(self, method, params):
        if method == "tab.new":
            assert params == {"adapter_name": "singleshot/ge"}
            return {"tab_id": "t"}
        return super().__call__(method, params)


def test_ge_requires_calibrated_pi_instead_of_custom_template(tmp_path):
    gui = GeGui()
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("singleshot_ge", {})
        assert reply.data["status"] == "needs_parameters", reply.data
        assert {item["parameter"] for item in reply.data["missing"]} == {"pi_ref"}
        assert not gui.ran
    finally:
        client.context.session.close()
