"""Drive recipe behavior through shipped tools and the GUI wire boundary."""

from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import pytest
from simpleeval import simple_eval

from ._recipe_support import LookbackGui, scalar, section
from ._support import make_client


@contextmanager
def recipe_client(tmp_path, respond):
    client = make_client(tmp_path, respond)
    try:
        yield client
    finally:
        client.context.session.close()


class TimeRabiGui(LookbackGui):
    def __init__(self, md=None):
        super().__init__()
        self.md = md or {}
        tree = self.publication["tree"]["children"]
        tree["reps"] = scalar(19)
        tree["sweep"] = section(
            length={
                "kind": "sweep",
                "valid": True,
                "inputs": {
                    key: scalar(value)["input"]
                    for key, value in {"start": 0.04, "stop": 0.8, "expts": 61}.items()
                },
            }
        )
        tree["modules"]["children"]["readout"]["ref"] = "<Custom:Pulse Readout>"
        tree["modules"]["children"]["qub_pulse"] = {
            "kind": "reference",
            "valid": True,
            "ref": "<Custom:Pulse>",
            "error": None,
            "children": {
                "freq": scalar(0.0),
                "gain": scalar(0.14),
                "waveform": section(length=scalar(1.0)),
            },
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

    def __call__(self, method, params):
        if method == "tab.new":
            assert params == {"adapter_name": "twotone/rabi/len_rabi"}
            return {"tab_id": "t"}
        result = super().__call__(method, params)
        if method == "tab.snapshot":
            result["tabs"][0]["adapter_name"] = "twotone/rabi/len_rabi"
        if method == "context.snapshot":
            result["ml"]["modules"] = {
                "drive": {"type": "pulse"},
                "calibrated": {"type": "readout/pulse"},
            }
        return result


def test_time_rabi_without_pi_uses_explicit_frequency_and_preserves_gui_start(tmp_path):
    gui = TimeRabiGui({"q_f": 6300.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        reply = client.call(
            "time_rabi",
            {
                "frequency_mhz": 6150.0,
                "gain": 0.21,
                "max_length_us": 1.5,
                "points": 71,
                "reps": 13,
                "rounds": 9,
            },
        )
        data = reply.data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["modules.qub_pulse.freq"]["value"] == 6150.0
        assert fields["modules.qub_pulse.freq"]["source"] == "frequency_mhz"
        assert fields["modules.qub_pulse.gain"]["value"] == 0.21
        assert fields["sweep.length"]["value"] == {
            "start": 0.04,
            "stop": 1.5,
            "expts": 71,
        }
        assert fields["reps"]["value"] == 13
        assert fields["rounds"]["value"] == 9
        assert fields["modules.reset"]["source"] == "disabled"
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["status"] == "finished"
        assert data["writeback"]["items"]
        assert reply.images
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert methods.index("tab.save_data") < methods.index("tab.analyze")


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

        if "qf_w" in self.md:
            inputs = self.publication["tree"]["children"]["sweep"]["children"]["freq"][
                "inputs"
            ]
            for key, sign in (("start", "-"), ("stop", "+")):
                expression = f"q_f {sign} 2.5 * qf_w"
                inputs[key] = {
                    **scalar(
                        simple_eval(expression, names=self.md)
                        if "q_f" in self.md
                        else None
                    )["input"],
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
                "drive": {"type": "pulse"},
                "reset": {"type": "reset/pulse"},
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


@pytest.mark.parametrize("reuse_tab_id", [None, "t"])
def test_twotone_reports_all_missing_frequency_sources_without_running(
    tmp_path, reuse_tab_id
):
    gui = DriveGui()
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("twotone_spectrum", {"reuse_tab_id": reuse_tab_id})
        methods = [method for method, _ in client.transport.sent]
        assert ("tab.reset_cfg" in methods) is (reuse_tab_id is not None)
        assert ("tab.new" in methods) is (reuse_tab_id is None)
        assert reply.data["status"] == "needs_parameters", reply.data
        assert {item["parameter"] for item in reply.data["missing"]} == {
            "center_mhz",
            "span_mhz",
            "readout_ref",
        }
        assert not gui.ran
    finally:
        client.context.session.close()


@pytest.mark.parametrize("span_mhz", [None, 80.0])
def test_twotone_explicit_center_removes_missing_calibration_dependency(
    tmp_path, span_mhz
):
    gui = DriveGui({"qf_w": 4.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        reply = client.call(
            "twotone_spectrum", {"center_mhz": 6300.0, "span_mhz": span_mhz}
        )
        assert reply.data["status"] == "finished", reply.data
        fields = reply.data["actual"]["fields"]
        assert fields["center_mhz"]["value"] == 6300.0
        assert fields["span_mhz"]["value"] == (20.0 if span_mhz is None else 80.0)


def test_twotone_preserves_valid_library_leaf_and_fills_missing_leaf(tmp_path):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0, "r_f": 7200.0})
    readout = gui.publication["tree"]["children"]["modules"]["children"]["readout"]
    readout["ref"] = "calibrated"
    readout["children"]["ro_cfg"]["children"]["ro_freq"] = scalar(None)
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("twotone_spectrum", {})
        assert reply.data["status"] == "finished", reply.data
        fields = reply.data["actual"]["fields"]
        assert fields["modules.readout.pulse_cfg.freq"]["value"] == 5000.0
        assert (
            fields["modules.readout.pulse_cfg.freq"]["source"] == "library:calibrated"
        )
        assert fields["modules.readout.ro_cfg.ro_freq"]["value"] == 7200.0
        assert fields["modules.readout.ro_cfg.ro_freq"]["source"] == "r_f"


@pytest.mark.parametrize("parameter", ["readout_ref", "drive_ref", "use_reset"])
def test_twotone_invalid_reference_stops_without_fallback(tmp_path, parameter):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0, "r_f": 7200.0})

    def respond(method, params):
        result = gui(method, params)
        if method == "tab.edit_cfg":
            for node in result["tree"]["children"]["modules"]["children"].values():
                if node.get("ref") == "unknown":
                    node["error"] = "Unknown library key"
                    result["status"] = "Invalid"
        return result

    with recipe_client(tmp_path, respond) as client:
        reply = client.call("twotone_spectrum", {parameter: "unknown"})
        assert reply.data["status"] == "failed", reply.data
        assert reply.data["error"]["reason"] == "invalid_cfg"
        assert not gui.ran
        assert sum(method == "tab.edit_cfg" for method, _ in client.transport.sent) == 1


def test_twotone_reuse_applies_explicit_drive_and_reset_once(tmp_path):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        reply = client.call(
            "twotone_spectrum",
            {"reuse_tab_id": "t", "drive_ref": "drive", "use_reset": "reset"},
        )
        assert reply.data["status"] == "finished", reply.data
        fields = reply.data["actual"]["fields"]
        assert fields["modules.qub_pulse"] == {"value": "drive", "source": "explicit"}
        assert fields["modules.reset"] == {"value": "reset", "source": "explicit"}
        methods = [method for method, _ in client.transport.sent]
        assert "tab.new" not in methods
        assert methods.count("tab.reset_cfg") == 1
        assert methods.count("tab.run_start") == 1


@pytest.mark.parametrize("failure", ["stale", "missing", "busy", "wrong_adapter"])
def test_twotone_reuse_failure_stops_without_retry_or_replacement(tmp_path, failure):
    gui = DriveGui({"q_f": 6100.0, "qf_w": 4.0, "r_f": 7200.0})
    with recipe_client(tmp_path, gui) as client:
        if failure == "stale":
            client.transport.replies["tab.reset_cfg"] = {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "stale_cfg",
                    "message": "changed",
                },
            }
        else:
            snapshot = gui("tab.snapshot", {"tab_id": "t"})
            if failure == "missing":
                snapshot["tabs"] = []
            elif failure == "busy":
                snapshot["tabs"][0]["interaction"]["is_analyzing"] = True
            else:
                snapshot["tabs"][0]["adapter_name"] = "other"
            client.transport.replies["tab.snapshot"] = {"ok": True, "result": snapshot}
        reply = client.call("twotone_spectrum", {"reuse_tab_id": "t"})
        assert reply.is_error
        assert reply.data["tab"] == "t"
        methods = [method for method, _ in client.transport.sent]
        assert "tab.new" not in methods
        assert "tab.run_start" not in methods
        assert methods.count("tab.reset_cfg") == int(failure == "stale")
