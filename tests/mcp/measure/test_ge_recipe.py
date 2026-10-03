"""GE calibration behavior through shipped tools and the GUI wire boundary."""

import base64
from copy import deepcopy
from typing import Any

import pytest

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

    def _observations(self):
        observations = super()._observations()
        observations["context.snapshot"]["ml"]["modules"] = self.library
        observations["tab.snapshot"]["tabs"][0]["adapter_name"] = "singleshot/ge"
        return observations

    def __call__(self, method, params) -> dict[str, Any]:
        self.calls.append((method, deepcopy(params)))
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
            return self._post_result(method, params)
        if method == "tab.new":
            assert params == {"adapter_name": "singleshot/ge"}
            return {"tab_id": "t"}
        return super().__call__(method, params)

    def _post_result(self, method, params):
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
        raise AssertionError(method)


@pytest.fixture()
def ge_client(tmp_path):
    gui = GeGui()
    client = make_client(tmp_path, gui)
    try:
        yield gui, client
    finally:
        client.context.session.close()


@pytest.mark.parametrize("reuse", [False, True])
def test_ge_reports_all_missing_calibration_without_running(ge_client, reuse):
    gui, client = ge_client
    gui.md.clear()
    data = client.call("singleshot_ge", {"reuse_tab_id": "t"} if reuse else {}).data
    assert data["status"] == "needs_parameters", data
    assert {item["parameter"] for item in data["missing"]} == {"pi_ref", "readout_ref"}
    assert not gui.ran
    methods = [method for method, _ in gui.calls]
    assert ("tab.reset_cfg" in methods) == reuse
    assert ("tab.new" in methods) != reuse


@pytest.mark.parametrize("explicit", [False, True])
def test_ge_preserves_calibrated_library_and_gui_shots_defaults(ge_client, explicit):
    gui, client = ge_client
    gui.md.clear()
    modules = gui.publication["tree"]["children"]["modules"]["children"]
    if not explicit:
        modules["probe_pulse"]["ref"] = "pi"
        modules["readout"]["ref"] = "readout"
    arguments = {"pi_ref": "pi", "readout_ref": "readout"} if explicit else {}
    data = client.call("singleshot_ge", arguments).data
    assert data["status"] == "finished", data
    fields = data["actual"]["fields"]
    assert fields["shots"]["value"] == 7000
    assert fields["shots"]["source"] == "gui_default"
    assert fields["modules.readout.ro_cfg.ro_freq"]["value"] == 5000.0
    assert fields["modules.readout.ro_cfg.ro_freq"]["source"] == "library:readout"
    assert fields["modules.probe_pulse"]["value"] == "pi"
    assert fields["modules.probe_pulse"]["source"] == (
        "explicit" if explicit else "gui_default"
    )


@pytest.mark.parametrize("reuse", [False, True])
def test_ge_optional_refs_are_explicit_and_omission_resets_them(ge_client, reuse):
    gui, client = ge_client
    gui.library.update(reset={}, init={})
    modules = gui.publication["tree"]["children"]["modules"]["children"]
    modules["reset"]["ref"] = "old_reset"
    modules["init_pulse"]["ref"] = "old_init"
    arguments = {
        "pi_ref": "pi",
        "reuse_tab_id": "t" if reuse else None,
        "use_reset": None if reuse else "reset",
        "init_pulse_ref": None if reuse else "init",
    }
    data = client.call("singleshot_ge", arguments).data
    assert data["status"] == "finished", data
    assert modules["reset"]["ref"] == (None if reuse else "reset")
    assert modules["init_pulse"]["ref"] == (None if reuse else "init")
    assert sum(method == "tab.run_start" for method, _ in gui.calls) == 1


@pytest.mark.parametrize(
    "parameter", ["pi_ref", "readout_ref", "use_reset", "init_pulse_ref"]
)
def test_ge_does_not_replace_missing_explicit_library_refs(ge_client, parameter):
    gui, client = ge_client
    data = client.call("singleshot_ge", {"pi_ref": "pi", parameter: "missing"}).data
    assert data["status"] == "failed", data
    assert data["error"]["reason"] == "invalid_cfg"
    assert not gui.ran


@pytest.mark.parametrize("failure", ["edit", "invalid"])
def test_ge_gui_cfg_rejection_stops_before_run(tmp_path, failure):
    gui = GeGui()

    def responder(method, params):
        result = gui(method, params)
        if method == "tab.edit_cfg" and failure == "invalid":
            result["status"] = "Invalid"
        return result

    client = make_client(tmp_path, responder)
    if failure == "edit":
        client.transport.replies["tab.edit_cfg"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "invalid_cfg",
                "message": "Reference is not applicable",
            },
        }
    try:
        data = client.call("singleshot_ge", {"pi_ref": "pi"}).data
        assert data["status"] == "failed", data
        assert data["error"]["reason"] == "invalid_cfg"
        assert not gui.ran
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    "arguments",
    [{"shots": value} for value in (0, -1, True, 2.5, float("inf"), float("nan"), "10")]
    + [
        {name: value}
        for name in (
            "reuse_tab_id",
            "readout_ref",
            "pi_ref",
            "use_reset",
            "init_pulse_ref",
        )
        for value in ("", " ", [], False)
    ],
)
def test_ge_rejects_invalid_arguments_before_preparing(tmp_path, arguments):
    gui = GeGui()
    client = make_client(tmp_path, gui)
    try:
        data = client.call("singleshot_ge", {"pi_ref": "pi", **arguments}).data
        assert data["status"] == "failed", data
        assert not gui.ran
        assert not any(method == "context.snapshot" for method, _ in gui.calls)
    finally:
        client.context.session.close()


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
