"""Drive recipe behavior through shipped tools and the GUI wire boundary."""

from copy import deepcopy
from typing import Any

import pytest
from simpleeval import simple_eval

from ._recipe_support import LookbackGui, scalar, section
from ._support import make_client


class DriveGui(LookbackGui):
    def __init__(self, md=None):
        super().__init__()
        self.md = md or {}
        self.publication["tree"]["children"]["modules"]["children"]["readout"][
            "ref"
        ] = "<Custom:Pulse Readout>"
        self.publication["tree"]["children"].update(
            reps=scalar(17),
            sweep=section(
                freq={
                    "kind": "sweep",
                    "valid": True,
                    "inputs": {
                        key: scalar(value)["input"]
                        for key, value in {
                            "start": 4500.0,
                            "stop": 5500.0,
                            "expts": 41,
                        }.items()
                    },
                }
            ),
        )
        self.publication["tree"]["children"]["modules"]["children"]["qub_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": None,
            "error": None,
            "children": {
                "freq": scalar(0.0),
                "gain": scalar(0.12),
                "waveform": section(length=scalar(2.0)),
            },
        }

        if "q_f" in self.md and "qf_w" in self.md:
            inputs = self.publication["tree"]["children"]["sweep"]["children"]["freq"][
                "inputs"
            ]
            for key, sign in (("start", "-"), ("stop", "+")):
                expression = f"q_f {sign} 2.5 * qf_w"
                inputs[key] = {
                    **scalar(simple_eval(expression, names=self.md))["input"],
                    "mode": "expression",
                    "raw": expression,
                }

    def _edit(self, params):
        ordinary = []
        for edit in params["edits"]:
            if edit["path"] == ["sweep", "freq"]:
                inputs = self.publication["tree"]["children"]["sweep"]["children"][
                    "freq"
                ]["inputs"]
                for key, value in edit["value"].items():
                    if isinstance(value, dict):
                        expression = value["__expr"]
                        inputs[key] = {
                            **scalar(simple_eval(expression, names=self.md))["input"],
                            "mode": "expression",
                            "raw": expression,
                        }
                    else:
                        inputs[key] = scalar(value)["input"]
            else:
                ordinary.append(edit)
        super()._edit({**params, "edits": ordinary})
        for edit in ordinary:
            if edit["path"] == ["modules", "readout"] and edit["value"] == {
                "__ref": "direct"
            }:
                readout = self.publication["tree"]["children"]["modules"]["children"][
                    "readout"
                ]
                readout["children"] = {"ro_freq": scalar(7100.0)}

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.new":
            assert params == {"adapter_name": "twotone/freq"}
            return {"tab_id": "t"}
        result = super().__call__(method, params)
        if method == "context.snapshot":
            result["ml"]["modules"] = {
                "calibrated": {"type": "readout/pulse"},
                "direct": {"type": "readout/direct"},
            }
        if method == "tab.snapshot":
            result = deepcopy(result)
            result["tabs"][0]["adapter_name"] = "twotone/freq"
        return result


@pytest.mark.parametrize("readout_ref", [None, "calibrated", "direct"])
def test_twotone_runs_once_with_gui_sweep_and_preserved_readout(tmp_path, readout_ref):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0, "r_f": 7200.0})
    client = make_client(tmp_path, gui)
    try:
        reply = client.call(
            "twotone_spectrum",
            {
                "readout_ref": readout_ref,
                "gain": 0.18,
                "pulse_length_us": 3.0,
                "points": 31,
                "reps": 23,
                "rounds": 7,
            },
        )
        data = reply.data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["sweep.freq"]["value"] == {
            "start": 6090.0,
            "stop": 6110.0,
            "expts": 31,
        }
        assert fields["modules.qub_pulse.gain"]["value"] == 0.18
        assert fields["modules.qub_pulse.waveform.length"]["value"] == 3.0
        assert fields["reps"]["value"] == 23
        assert fields["rounds"]["value"] == 7
        assert fields["modules.reset"]["source"] == "disabled"
        if readout_ref == "direct":
            assert fields["modules.readout.ro_freq"]["value"] == 7100.0
        else:
            expected = 5000.0 if readout_ref else 7200.0
            assert fields["modules.readout.pulse_cfg.freq"]["value"] == expected
            assert fields["modules.readout.ro_cfg.ro_freq"]["value"] == expected
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


@pytest.mark.parametrize(
    "arguments",
    [
        {"center_mhz": True},
        {"center_mhz": float("nan")},
        {"span_mhz": float("inf")},
        {"span_mhz": 0.0},
        {"span_mhz": -1.0},
        {"gain": False},
        {"pulse_length_us": "bad"},
        {"points": 2.5},
        {"points": True},
        {"reps": 1.5},
        {"rounds": False},
        {"readout_ref": ""},
        {"drive_ref": 1},
        {"use_reset": True},
        {"reuse_tab_id": " "},
    ],
)
def test_twotone_rejects_invalid_explicit_values_before_preparation(
    tmp_path, arguments
):
    gui = DriveGui()
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("twotone_spectrum", arguments)
        assert reply.data["status"] == "failed", reply.data
        assert reply.data["error"]["phase"] == "preparing"
        assert not any(
            method == "context.snapshot" for method, _ in client.transport.sent
        )
        assert not gui.ran
    finally:
        client.context.session.close()


def test_twotone_requires_calibrated_readout_instead_of_inline_template(tmp_path):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0})
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("twotone_spectrum", {})
        assert reply.data["status"] == "needs_parameters", reply.data
        assert {item["parameter"] for item in reply.data["missing"]} == {"readout_ref"}
        assert not gui.ran
    finally:
        client.context.session.close()


def test_twotone_reports_all_missing_frequency_sources_without_running(tmp_path):
    gui = DriveGui()
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("twotone_spectrum", {})
        assert reply.data["status"] == "needs_parameters", reply.data
        assert {item["parameter"] for item in reply.data["missing"]} == {
            "center_mhz",
            "span_mhz",
            "readout_ref",
        }
        assert not gui.ran
    finally:
        client.context.session.close()
