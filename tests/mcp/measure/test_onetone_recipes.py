"""Onetone promises through the shipped recipe tools and recording GUI."""

import base64
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from simpleeval import NameNotDefined, simple_eval
from zcu_tools.mcp.core.reply import ToolReply

from ._recipe_support import PNG, LookbackGui, scalar, section
from ._support import make_client


@contextmanager
def recipe_client(tmp_path, respond):
    client = make_client(tmp_path, respond)
    try:
        yield client
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    "arguments",
    [
        {"center_mhz": True},
        {"center_mhz": float("nan")},
        {"span_mhz": 0},
        {"span_mhz": -1},
        {"span_mhz": float("inf")},
        {"gain": False},
        {"gain": "bad"},
        {"points": 2.5},
        {"points": True},
        {"reps": 1.5},
        {"rounds": False},
        {"reuse_tab_id": ""},
        {"readout_ref": 23},
    ],
)
def test_spectrum_rejects_explicit_invalid_input_before_gui_work(tmp_path, arguments):
    gui = OnetoneGui({"r_f": 6000.0, "rf_w": 4.0})
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("onetone_spectrum", arguments)
        assert isinstance(reply, ToolReply)
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["phase"] == "preparing"
        assert not gui.ran
        assert not any(
            method == "context.snapshot" for method, _ in client.transport.sent
        )


def test_flux_reports_all_missing_sources_in_one_handoff(tmp_path):
    gui = OnetoneGui(experiment="onetone/flux_dep")

    def respond(method, params):
        if method == "value.list":
            return {"values": []}
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        reply = client.call("onetone_spectrum_over_flux", {})
        assert isinstance(reply, ToolReply)
        assert reply.data["status"] == "needs_parameters", reply.data
        assert {item["parameter"] for item in reply.data["missing"]} == {
            "center_mhz",
            "span_mhz",
            "flux_device",
            "flux_range",
        }
        assert not gui.ran
        assert not reply.is_error


@pytest.mark.parametrize(
    "arguments",
    [
        {"flux_device": ""},
        {"flux_device": False},
        {"flux_device": " "},
        {"freq_points": True},
        {"freq_points": 1.5},
        {"flux_points": False},
        {"flux_range": [0, True]},
        {"flux_range": [0, float("inf")]},
        {"flux_range": [0]},
        {"flux_range": [0, 1, 2]},
        {"flux_range": {"start": 0, "stop": 1}},
        {"flux_range": "0,1"},
    ],
)
def test_flux_rejects_invalid_explicit_inputs_instead_of_missing_handoff(
    tmp_path, arguments
):
    gui = OnetoneGui(experiment="onetone/flux_dep")
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("onetone_spectrum_over_flux", arguments)
        assert isinstance(reply, ToolReply)
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert not any(
            method == "context.snapshot" for method, _ in client.transport.sent
        )


@pytest.mark.parametrize("stage", ["tab.edit_cfg", "tab.run_start"])
def test_spectrum_source_change_fails_without_retry_or_blind_scan(tmp_path, stage):
    gui = OnetoneGui({"r_f": 6100.0, "rf_w": 4.0})

    with recipe_client(tmp_path, gui) as client:
        client.transport.replies[stage] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "stale",
                "message": "Source changed",
            },
        }
        reply = client.call("onetone_spectrum", {})
        assert isinstance(reply, ToolReply)
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["reason"] == "stale"
        assert not gui.ran
        assert sum(method == stage for method, _ in client.transport.sent) == 1


class OnetoneGui(LookbackGui):
    def __init__(self, md=None, experiment="onetone/freq"):
        super().__init__()
        self.experiment = experiment
        self.md = md or {}
        center = "r_f" if "r_f" in self.md else "5000.0"
        width = "2.5 * rf_w" if "rf_w" in self.md else "500.0"
        self.publication["tree"]["children"].update(
            reps=scalar(17),
            sweep=section(
                freq={
                    "kind": "sweep",
                    "valid": True,
                    "inputs": {
                        "start": self.input(f"{center} - {width}" if md else 4500.0),
                        "stop": self.input(f"{center} + {width}" if md else 5500.0),
                        "expts": self.input(41),
                        "step": self.input(25.0),
                    },
                },
            ),
        )
        readout = self.publication["tree"]["children"]["modules"]["children"]["readout"]
        readout["children"]["pulse_cfg"]["children"]["gain"] = scalar(0.21)

    def input(self, value):
        if not isinstance(value, str):
            return scalar(value)["input"]
        try:
            resolved = simple_eval(value, names=self.md)
            error = None
        except (TypeError, NameNotDefined) as exc:
            resolved, error = None, str(exc)
        return {
            "mode": "expression",
            "raw": value,
            "resolved": resolved,
            "error": error,
            "validation_error": None,
        }

    def _edit(self, params):
        ordinary = []
        for edit in params["edits"]:
            if edit["path"][0] == "sweep":
                inputs = self.publication["tree"]["children"]["sweep"]["children"][
                    edit["path"][1]
                ]["inputs"]
                for key, value in edit["value"].items():
                    inputs[key] = self.input(
                        value["__expr"] if isinstance(value, dict) else value
                    )
            else:
                ordinary.append(edit)
        super()._edit({**params, "edits": ordinary})

    def __call__(self, method, params):
        if method == "value.list":
            return {"values": []}
        if method == "tab.new":
            assert params == {"adapter_name": self.experiment}
            return {"tab_id": "t"}
        reply = super().__call__(method, params)
        if method == "tab.snapshot":
            reply["tabs"][0]["adapter_name"] = self.experiment
        return reply


class PowerGui(OnetoneGui):
    def __init__(self):
        super().__init__({"r_f": 6100.0, "rf_w": 4.0}, "onetone/power_dep")
        self.publication["tree"]["children"]["sweep"]["children"]["gain"] = {
            "kind": "sweep", "valid": True,
            "inputs": {key: self.input(value) for key, value in {
                "start": 0.03, "stop": 0.27, "expts": 13, "step": 0.02,
            }.items()},
        }

    def __call__(self, method, params):
        if method == "tab.get_figure":
            assert self.raw_saved
            assert params == {"tab_id": "t", "subtab_id": "run", "run_operation_id": 71}
            return {"png_b64": base64.b64encode(PNG).decode()}
        return super().__call__(method, params)


@pytest.mark.parametrize("arguments, expected_gain", [
    ({}, {"start": 0.03, "stop": 0.27, "expts": 13}),
    ({"reuse_tab_id": "t", "gain_range": [0.1, 0.7], "gain_points": 7, "freq_points": 23},
     {"start": 0.1, "stop": 0.7, "expts": 7}),
])
def test_power_saves_raw_and_delivers_only_a_run_preview(tmp_path, arguments, expected_gain):
    gui = PowerGui()
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("onetone_spectrum_over_power", arguments)
        assert isinstance(reply, ToolReply)
        data = reply.data
        assert data["status"] == "finished", data
        assert data["analysis_mode"] == "none"
        assert data["analysis"] is None
        assert data["writeback"] is None
        assert data["actual"]["fields"]["sweep.gain"]["value"] == expected_gain
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["preview"]["kind"] == "run_preview"
        assert Path(data["preview"]["path"]).read_bytes() == PNG
        assert reply.images[0].data == PNG
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert "tab.analyze" not in methods
        assert "tab.writeback_preview" not in methods
        assert methods.index("tab.save_data") < methods.index("tab.get_figure")


class FluxGui(OnetoneGui):
    def __init__(self):
        super().__init__(
            {"r_f": 6100.0, "rf_w": 4.0, "flx_half": 0.001, "flx_int": 0.003},
            experiment="onetone/flux_dep",
        )
        root = self.publication["tree"]["children"]
        root["dev"] = section(flux_dev=scalar("flux_yoko"))
        root["sweep"]["children"]["flux"] = {
            "kind": "sweep",
            "valid": True,
            "inputs": {
                "start": self.input("2 * flx_int - flx_half"),
                "stop": self.input("2 * flx_half - flx_int"),
                "expts": self.input(19),
                "step": self.input(-0.006 / 18),
            },
        }

    def __call__(self, method, params):
        if method == "value.list":
            return {"values": [{"key": "device.flux.name"}]}
        if method == "value.read":
            assert params == {"key": "device.flux.name"}
            return {"key": "device.flux.name", "value": "coil"}
        if method == "device.list":
            return {"devices": [{"name": "coil"}, {"name": "alternate"}]}
        if method == "device.snapshot":
            assert params["name"] in ("coil", "alternate")
            return {
                "snapshot": {
                    "name": params["name"],
                    "unit": "A" if params["name"] == "coil" else "V",
                }
            }
        return super().__call__(method, params)


@pytest.mark.parametrize("explicit", [False, True])
def test_flux_saves_one_survey_with_physical_device_and_actual_conditions(
    tmp_path, explicit
):
    gui = FluxGui()
    arguments: dict[str, Any] = {
        "readout_ref": "calibrated",
        "freq_points": 31,
        "reps": 23,
        "rounds": 7,
        "gain": 0.13,
    }
    if explicit:
        arguments.update(
            reuse_tab_id="t",
            flux_device="alternate",
            flux_range=[-0.003, 0.007],
            flux_points=11,
        )
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("onetone_spectrum_over_flux", arguments)
        assert isinstance(reply, ToolReply)
        data = reply.data
        assert data["status"] == "finished", data
        fields = data["actual"]["fields"]
        assert fields["dev.flux_dev"] == {
            "value": "alternate" if explicit else "coil",
            "unit": "V" if explicit else "A",
            "source": "explicit" if explicit else "device.flux.name",
        }
        assert fields["sweep.flux"]["value"] == {
            "start": -0.003 if explicit else 0.005,
            "stop": 0.007 if explicit else -0.001,
            "expts": 11 if explicit else 19,
        }
        assert fields["sweep.flux"]["source"] == (
            "explicit" if explicit else "gui_calibration"
        )
        assert fields["sweep.freq"]["value"]["expts"] == 31
        assert fields["modules.readout"]["value"] == "calibrated"
        assert fields["modules.readout.pulse_cfg.gain"]["value"] == 0.13
        assert fields["reps"]["value"] == 23
        assert fields["rounds"]["value"] == 7
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["status"] == "finished"
        assert data["writeback"]["items"]
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert ("tab.reset_cfg" in methods) is explicit
        assert ("value.read" in methods) is not explicit


@pytest.mark.parametrize("reuse_tab_id", [None, "t"])
def test_onetone_reports_missing_frequency_without_running(tmp_path, reuse_tab_id):
    gui = OnetoneGui()
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("onetone_spectrum", {"reuse_tab_id": reuse_tab_id})
        assert isinstance(reply, ToolReply)
        assert reply.data["status"] == "needs_parameters"
        assert {item["parameter"] for item in reply.data["missing"]} == {
            "center_mhz",
            "span_mhz",
        }
        assert not reply.is_error
        assert not gui.ran
        methods = [method for method, _ in client.transport.sent]
        assert ("tab.reset_cfg" in methods) is (reuse_tab_id is not None)
        assert ("tab.new" in methods) is (reuse_tab_id is None)


@pytest.mark.parametrize(
    "md, arguments, start, stop, center_source, span_source",
    [
        ({"r_f": 6100.0, "rf_w": 4.0}, {}, 6090.0, 6110.0, "r_f", "gui_linewidth"),
        (
            {"rf_w": 4.0},
            {"center_mhz": 6200.0},
            6190.0,
            6210.0,
            "explicit",
            "gui_linewidth",
        ),
        (
            {"r_f": None, "rf_w": 4.0},
            {"center_mhz": 6200.0},
            6190.0,
            6210.0,
            "explicit",
            "gui_linewidth",
        ),
        ({"r_f": 6100.0}, {"span_mhz": 8.0}, 6096.0, 6104.0, "r_f", "explicit"),
        (
            {},
            {"center_mhz": 6200.0, "span_mhz": 16.0},
            6192.0,
            6208.0,
            "explicit",
            "explicit",
        ),
        (
            {"r_f": 6100.0, "rf_w": 4.0},
            {"center_mhz": 6200.0},
            6190.0,
            6210.0,
            "explicit",
            "gui_linewidth",
        ),
        (
            {"r_f": 6100.0, "rf_w": 4.0},
            {"span_mhz": 8.0},
            6096.0,
            6104.0,
            "r_f",
            "explicit",
        ),
    ],
)
def test_spectrum_saves_one_run_with_gui_derived_frequency_and_averages(
    tmp_path, md, arguments, start, stop, center_source, span_source
):
    gui = OnetoneGui(md)
    before = deepcopy(gui.publication)
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("onetone_spectrum", arguments)
        assert isinstance(reply, ToolReply)
        data = reply.data
        assert data["status"] == "finished", data
        assert not reply.is_error
        fields = data["actual"]["fields"]
        assert fields["sweep.freq"]["value"] == {
            "start": start,
            "stop": stop,
            "expts": 41,
        }
        assert fields["center_mhz"]["source"] == center_source
        assert fields["span_mhz"]["source"] == span_source
        assert fields["reps"]["value"] == 17
        assert fields["rounds"]["value"] == 3
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["status"] == "finished"
        assert data["writeback"]["items"][0]["proposed"] == 0.24
        assert reply.images
        assert before["cfg_ref"] != data["actual"]["cfg_ref"]
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == 1
        assert methods.index("tab.save_data") < methods.index("tab.analyze")
