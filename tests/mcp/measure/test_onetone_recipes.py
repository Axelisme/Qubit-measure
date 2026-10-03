"""Onetone promises through the shipped recipe tools and recording GUI."""

from copy import deepcopy

import pytest
from simpleeval import simple_eval
from zcu_tools.mcp.core.reply import ToolReply

from ._recipe_support import LookbackGui, scalar, section
from ._support import make_client


class OnetoneGui(LookbackGui):
    def __init__(self, md=None):
        super().__init__()
        self.md = md or {}
        self.publication["tree"]["children"].update(
            reps=scalar(17),
            sweep=section(
                freq={
                    "kind": "sweep",
                    "valid": True,
                    "inputs": {
                        "start": self.input("r_f - 2.5 * rf_w" if md else 4500.0),
                        "stop": self.input("r_f + 2.5 * rf_w" if md else 5500.0),
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
        return {
            "mode": "expression",
            "raw": value,
            "resolved": simple_eval(value, names=self.md),
            "error": None,
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
            assert params == {"adapter_name": "onetone/freq"}
            return {"tab_id": "t"}
        reply = super().__call__(method, params)
        if method == "tab.snapshot":
            reply["tabs"][0]["adapter_name"] = "onetone/freq"
        return reply


@pytest.mark.parametrize("reuse_tab_id", [None, "t"])
def test_onetone_reports_missing_frequency_without_running(tmp_path, reuse_tab_id):
    gui = OnetoneGui()
    client = make_client(tmp_path, gui)
    try:
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
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    "md, arguments, start, stop, center_source, span_source",
    [
        ({"r_f": 6100.0, "rf_w": 4.0}, {}, 6090.0, 6110.0, "r_f", "gui_linewidth"),
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
    client = make_client(tmp_path, gui)
    try:
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
    finally:
        client.context.session.close()
