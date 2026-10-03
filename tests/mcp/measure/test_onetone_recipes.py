"""Onetone promises through the shipped recipe tools and recording GUI."""

from contextlib import contextmanager
from copy import deepcopy

import pytest
from simpleeval import NameNotDefined, simple_eval
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_support import LookbackGui, scalar, section
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


@pytest.mark.parametrize("arguments", [
    {"flux_device": ""}, {"flux_device": False}, {"flux_device": " "},
    {"freq_points": True}, {"freq_points": 1.5}, {"flux_points": False},
    {"flux_range": [0, True]}, {"flux_range": [0, float("inf")]},
    {"flux_range": [0]}, {"flux_range": [0, 1, 2]},
    {"flux_range": {"start": 0, "stop": 1}}, {"flux_range": "0,1"},
])
def test_flux_rejects_invalid_explicit_inputs_instead_of_missing_handoff(tmp_path, arguments):
    gui = OnetoneGui(experiment="onetone/flux_dep")
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("onetone_spectrum_over_flux", arguments)
        assert isinstance(reply, ToolReply)
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert not any(method == "context.snapshot" for method, _ in client.transport.sent)


@pytest.mark.parametrize("stage", ["tab.edit_cfg", "tab.run_start"])
def test_spectrum_source_change_fails_without_retry_or_blind_scan(tmp_path, stage):
    gui = OnetoneGui({"r_f": 6100.0, "rf_w": 4.0})
    def respond(method, params):
        if method == stage:
            raise GuiRpcError("Source changed", reason="stale")
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
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
            if edit["path"] == ["sweep", "freq"]:
                inputs = self.publication["tree"]["children"]["sweep"]["children"][
                    "freq"
                ]["inputs"]
                for key, value in edit["value"].items():
                    inputs[key] = self.input(
                        value["__expr"] if isinstance(value, dict) else value
                    )
            else:
                ordinary.append(edit)
        super()._edit({**params, "edits": ordinary})

    def __call__(self, method, params):
        if method == "tab.new":
            assert params == {"adapter_name": self.experiment}
            return {"tab_id": "t"}
        reply = super().__call__(method, params)
        if method == "tab.snapshot":
            reply["tabs"][0]["adapter_name"] = self.experiment
        return reply


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
        ({"rf_w": 4.0}, {"center_mhz": 6200.0}, 6190.0, 6210.0, "explicit", "gui_linewidth"),
        ({"r_f": None, "rf_w": 4.0}, {"center_mhz": 6200.0}, 6190.0, 6210.0, "explicit", "gui_linewidth"),
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
