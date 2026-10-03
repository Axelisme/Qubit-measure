"""Coherence behavior through shipped tools and the GUI wire boundary."""

from typing import Any

import pytest

from ._recipe_support import LookbackGui, scalar, section
from ._support import make_client


class CoherenceGui(LookbackGui):
    def __init__(self, pi_ref="<Custom:Pulse>"):
        super().__init__()
        self.md = {"r_f": 5100.0}
        self.library = {"pi": {"type": "pulse"}}
        tree = self.publication["tree"]["children"]
        tree["reps"] = scalar(19)
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
            assert params == {"adapter_name": "twotone/t1"}
            return {"tab_id": "t"}
        result = super().__call__(method, params)
        if method == "tab.snapshot":
            result["tabs"][0]["adapter_name"] = "twotone/t1"
        if method == "context.snapshot":
            result["ml"]["modules"] = self.library
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
