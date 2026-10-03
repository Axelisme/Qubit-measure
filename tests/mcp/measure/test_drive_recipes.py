"""Drive recipe behavior through shipped tools and the GUI wire boundary."""

from copy import deepcopy
from typing import Any

from ._recipe_support import LookbackGui, scalar, section
from ._support import make_client


class DriveGui(LookbackGui):
    def __init__(self):
        super().__init__()
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

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.new":
            assert params == {"adapter_name": "twotone/freq"}
            return {"tab_id": "t"}
        result = super().__call__(method, params)
        if method == "tab.snapshot":
            result = deepcopy(result)
            result["tabs"][0]["adapter_name"] = "twotone/freq"
        return result


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
