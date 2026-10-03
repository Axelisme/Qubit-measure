"""Coherence behavior through shipped tools and the GUI wire boundary."""

from contextlib import contextmanager
from typing import Any

import pytest

from ._recipe_support import LookbackGui, scalar, section
from ._support import make_client


@contextmanager
def recipe_client(tmp_path, gui):
    client = make_client(tmp_path, gui)
    try:
        yield client
    finally:
        client.context.session.close()


class CoherenceGui(LookbackGui):
    def __init__(self, pi_ref="<Custom:Pulse>", adapter="t1"):
        super().__init__()
        self.adapter = adapter
        self.md = {"r_f": 5100.0}
        self.library = {"pi": {"type": "pulse"}, "pi2": {"type": "pulse"}}
        tree = self.publication["tree"]["children"]
        tree["reps"] = scalar(19)
        tree["detune_ratio"] = scalar(0.2)
        tree["modules"]["children"]["pi2_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": "<Custom:Pulse>",
            "error": None,
            "children": {"freq": scalar(6100.0)},
        }
        tree["sweep"] = section(
            length={
                "kind": "sweep",
                "valid": True,
                "inputs": {
                    key: scalar(value)["input"]
                    for key, value in {"start": 0.04, "stop": 60.0, "expts": 61}.items()
                },
            }
        )
        self.publication["tree"]["children"]["modules"]["children"]["pi_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": pi_ref,
            "error": None,
            "children": {"freq": scalar(6100.0)},
        }

    def _edit(self, params):
        ordinary = []
        for edit in params["edits"]:
            if edit["path"] == ["sweep", "length"]:
                inputs = self.publication["tree"]["children"]["sweep"]["children"][
                    "length"
                ]["inputs"]
                inputs.update(
                    {
                        key: scalar(value)["input"]
                        for key, value in edit["value"].items()
                    }
                )
            else:
                ordinary.append(edit)
        super()._edit({**params, "edits": ordinary})

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.new":
            assert params == {"adapter_name": f"twotone/{self.adapter}"}
            return {"tab_id": "t"}
        result = super().__call__(method, params)
        if method == "tab.snapshot":
            result["tabs"][0]["adapter_name"] = f"twotone/{self.adapter}"
        if method == "context.snapshot":
            result["ml"]["modules"] = self.library
        return result


@pytest.mark.parametrize(
    "recipe,parameter",
    [
        ("t1", "pi_ref"),
        ("t2ramsey", "pi2_ref"),
        ("t2echo", "pi_ref"),
        ("t2echo", "pi2_ref"),
        ("t1", "readout_ref"),
        ("t1", "use_reset"),
    ],
)
def test_coherence_explicit_reference_must_be_a_library_entry(
    tmp_path, recipe, parameter
):
    gui = CoherenceGui(pi_ref="pi", adapter=recipe)
    with recipe_client(tmp_path, gui) as client:
        data = client.call(recipe, {parameter: "<Custom:Pulse>"}).data
        assert data["status"] == "failed", data
        assert data["error"]["reason"] == "invalid_cfg"
        assert data["tab"] == "t"
        assert not gui.ran


@pytest.mark.parametrize("recipe", ["t2ramsey", "t2echo"])
def test_t2_runs_with_total_delay_and_unchanged_detune_units(tmp_path, recipe):
    gui = CoherenceGui(adapter=recipe)
    arguments = {
        "pi2_ref": "pi2",
        "max_delay_us": 84.0,
        "points": 43,
        "detune_ratio": 0.37,
        "reps": 7,
        "rounds": 5,
    }
    if recipe == "t2echo":
        arguments["pi_ref"] = "pi"
    with recipe_client(tmp_path, gui) as client:
        reply = client.call(recipe, arguments)
        data = reply.data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["sweep.length"]["value"] == {
            "start": 0.04,
            "stop": 84.0,
            "expts": 43,
        }
        assert fields["detune_ratio"]["value"] == 0.37
        assert fields["modules.pi2_pulse"]["value"] == "pi2"
        if recipe == "t2echo":
            assert fields["modules.pi_pulse"]["value"] == "pi"
        assert fields["modules.reset"]["value"] is None
        assert fields["reps"]["value"] == 7
        assert fields["rounds"]["value"] == 5
        assert data["tab"] == "t"
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["status"] == "finished"
        assert reply.images
        assert [method for method, _ in client.transport.sent].count(
            "tab.run_start"
        ) == 1


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


@pytest.mark.parametrize("reuse_tab_id", [None, "t"])
def test_t1_selected_library_pulse_preserves_gui_delay_defaults(tmp_path, reuse_tab_id):
    gui = CoherenceGui(pi_ref="pi")
    with recipe_client(tmp_path, gui) as client:
        data = client.call("t1", {"reuse_tab_id": reuse_tab_id}).data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["modules.pi_pulse"] == {"value": "pi", "source": "gui_default"}
        assert fields["sweep.length"]["value"] == {
            "start": 0.04,
            "stop": 60.0,
            "expts": 61,
        }
        assert fields["sweep.length"]["source"]["stop"] == "gui_default"
        assert fields["modules.readout.pulse_cfg.freq"]["input"]["raw"] == "r_f"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.reset_cfg") == (1 if reuse_tab_id else 0)
        assert methods.count("tab.new") == (0 if reuse_tab_id else 1)
        assert methods.count("tab.run_start") == 1


@pytest.mark.parametrize("arguments", [{}, {"pi_ref": "pi"}])
def test_t1_reports_all_missing_calibration_sources(tmp_path, arguments):
    gui = CoherenceGui()
    gui.md = {}
    with recipe_client(tmp_path, gui) as client:
        data = client.call("t1", arguments).data
        assert data["status"] == "needs_parameters", data
        missing = {item["parameter"] for item in data["missing"]}
        assert missing == (
            {"readout_ref", "pi_ref"} if not arguments else {"readout_ref"}
        )
        assert data["tab"] == "t"
        assert not gui.ran


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_delay_us": True},
        {"max_delay_us": float("nan")},
        {"max_delay_us": float("inf")},
        {"max_delay_us": 0},
        {"max_delay_us": -1},
        {"points": 1},
        {"points": 2.5},
        {"points": True},
        {"reps": False},
        {"rounds": 1.5},
        {"pi_ref": " "},
        {"readout_ref": 7},
        {"use_reset": False},
        {"reuse_tab_id": ""},
    ],
)
def test_t1_rejects_invalid_inputs_before_preparing(tmp_path, arguments):
    gui = CoherenceGui(pi_ref="pi")
    client = make_client(tmp_path, gui)
    try:
        data = client.call("t1", arguments).data
        assert data["status"] == "failed", data
        assert not gui.ran
        assert not any(
            method == "context.snapshot" for method, _ in client.transport.sent
        )
    finally:
        client.context.session.close()


def test_t1_runs_once_with_calibrated_pi_and_explicit_delay(tmp_path):
    gui = CoherenceGui()
    client = make_client(tmp_path, gui)
    try:
        reply = client.call(
            "t1",
            {
                "pi_ref": "pi",
                "max_delay_us": 80.0,
                "points": 81,
                "reps": 13,
                "rounds": 9,
            },
        )
        data = reply.data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["modules.pi_pulse"]["value"] == "pi"
        assert fields["modules.reset"]["value"] is None
        assert fields["modules.reset"]["source"] == "disabled"
        assert fields["sweep.length"]["value"] == {
            "start": 0.04,
            "stop": 80.0,
            "expts": 81,
        }
        assert fields["reps"]["value"] == 13
        assert fields["rounds"]["value"] == 9
        assert fields["modules.readout.pulse_cfg.freq"]["value"] == 5100.0
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["status"] == "finished"
        assert data["writeback"]["items"]
        assert reply.images
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert methods.index("tab.save_data") < methods.index("tab.analyze")
    finally:
        client.context.session.close()
