"""GE calibration behavior through shipped tools and the GUI wire boundary."""

import base64
from copy import deepcopy

from ._recipe_support import PNG, LookbackGui, scalar
from ._support import make_client


class GeGui(LookbackGui):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.library = {"pi": {}, "readout": {}}
        self.md = {"r_f": 5100.0}
        self.publication["tree"]["children"]["shots"] = scalar(7000)
        self.publication["tree"]["children"]["reps"] = scalar(1)
        self.publication["tree"]["children"]["rounds"] = scalar(1)
        modules = self.publication["tree"]["children"]["modules"]["children"]
        modules["probe_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": "<Custom:Pulse>",
            "error": None,
            "children": {"freq": scalar(6100.0)},
        }

    def __call__(self, method, params):
        self.calls.append((method, deepcopy(params)))
        if method == "context.snapshot":
            return {"md": self.md, "ml": {"modules": self.library}}
        if method == "tab.post_analyze":
            assert params == {
                "tab_id": "t",
                "updates": {},
                "operation_id": 93,
                "run_operation_id": 71,
            }
            return {
                "operation_id": 104,
                "interactive": False,
                "params": {"bins": 64},
                "invalidated_on_success": [],
            }
        if method == "operation.await" and params["operation_id"] == 104:
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_post_analyze_result":
            assert params == {"tab_id": "t", "operation_id": 104}
            return {
                "summary": {"fidelity": 0.98},
                "params": {"bins": 64},
                "operation_state": {"post_analysis_state": {"figure_names": ["cloud"]}},
            }
        if params.get("operation_id") == 104:
            assert params["subtab_id"] == "post_analysis"
            if method == "tab.save_image":
                assert params["figure_name"] == "cloud"
                return {"image_path": "/actual/cloud.png"}
            if method == "tab.get_figure":
                return {"png_b64": base64.b64encode(PNG).decode()}
            if method == "tab.writeback_preview":
                return {
                    "has_draft": True,
                    "items": [{"id": "classifier", "proposed": 0.98}],
                }
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


def test_ge_saves_and_delivers_primary_then_post_without_rerun(tmp_path):
    gui = GeGui()
    modules = gui.publication["tree"]["children"]["modules"]["children"]
    modules["reset"]["ref"] = "old_reset"
    modules["init_pulse"]["ref"] = "old_init"
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("singleshot_ge", {"pi_ref": "pi", "shots": 1234})
        data = reply.data
        assert data["status"] == "finished", data
        assert data["analysis_mode"] == "primary_post"
        assert data["analysis_stage"] == "post"
        assert data["analysis"]["result"]["summary"] == {"offset": 0.24}
        assert data["post_analysis"]["result"]["summary"] == {"fidelity": 0.98}
        assert data["analysis"]["saved_images"] == [
            {"figure_name": "trace", "image_path": "/actual/trace.png"}
        ]
        assert data["post_analysis"]["saved_images"] == [
            {"figure_name": "cloud", "image_path": "/actual/cloud.png"}
        ]
        assert data["writeback"]["items"][0]["id"] == "md-1"
        assert data["post_writeback"]["items"][0]["id"] == "classifier"
        assert len(reply.images) == 2
        assert data["actual"]["fields"]["shots"]["value"] == 1234
        assert modules["probe_pulse"]["ref"] == "pi"
        assert modules["reset"]["ref"] is None
        assert modules["init_pulse"]["ref"] is None
        stages = {
            "tab.run_start",
            "tab.save_data",
            "tab.analyze",
            "tab.post_analyze",
            "tab.save_image",
            "tab.get_figure",
            "tab.writeback_preview",
        }
        assert [method for method, _ in gui.calls if method in stages] == [
            "tab.run_start",
            "tab.save_data",
            "tab.analyze",
            "tab.save_image",
            "tab.get_figure",
            "tab.writeback_preview",
            "tab.post_analyze",
            "tab.save_image",
            "tab.get_figure",
            "tab.writeback_preview",
        ]
    finally:
        client.context.session.close()
