"""Recording onetone GUI responses and recipe session lifecycle helpers."""

import base64
from contextlib import contextmanager
from typing import Any

from pydantic import ValidationError
from zcu_tools.gui.session.expression import ExpressionError
from zcu_tools.mcp.core.reply import ToolReply

from ._recipe_support import PNG, LookbackGui, scalar, section
from ._support import MeasureClient, make_client


@contextmanager
def recipe_client(tmp_path, respond):
    client = make_client(tmp_path, respond)
    try:
        yield client
    finally:
        client.context.session.close()


def skip_writeback(client: MeasureClient, question: ToolReply) -> ToolReply:
    """Answer an observed production question without any draft writes."""
    assert question.data["status"] == "awaiting_answer", question.data
    assert tuple(question.data["question_items"]) == ("predict_offset",)
    before = list(client.transport.sent)
    reply = client.call(
        "answer", {"recipe": question.data["execution"], "decision": "skipped"}
    )
    assert isinstance(reply, ToolReply) and reply.data["status"] == "finished"
    assert client.transport.sent == before
    return reply


class OnetoneGui(LookbackGui):
    def __init__(self, md=None, experiment="onetone/freq"):
        super().__init__()
        self.experiment = experiment
        self.md = md or {}
        center = "$resonator.freq" if "$resonator.freq" in self.md else "5000.0"
        width = "2.5 * $resonator.kappa" if "$resonator.kappa" in self.md else "500.0"
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
            resolved = self.evaluate_expression(value)
            error = None
        except (ValidationError, ExpressionError) as exc:
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

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "value.list":
            return {"values": []}
        if method == "tab.new":
            assert params == {"adapter_name": self.experiment}
            return {"tab_id": "t"}
        reply = super().__call__(method, params)
        if method == "tab.snapshot":
            reply["tabs"][0]["adapter_name"] = self.experiment
        if method == "tab.writeback_preview":
            reply["items"][0].update(kind="parameter", target_name="predict_offset")
        return reply


class PowerGui(OnetoneGui):
    def __init__(self):
        super().__init__(
            {"$resonator.freq": 6100.0, "$resonator.kappa": 4.0}, "onetone/power_dep"
        )
        self.publication["tree"]["children"]["sweep"]["children"]["gain"] = {
            "kind": "sweep",
            "valid": True,
            "inputs": {
                key: self.input(value)
                for key, value in {
                    "start": 0.03,
                    "stop": 0.27,
                    "expts": 13,
                    "step": 0.02,
                }.items()
            },
        }

    def __call__(self, method, params):
        if method == "tab.get_figure":
            assert self.raw_saved
            assert params == {"tab_id": "t", "subtab_id": "run", "run_operation_id": 71}
            return {"png_b64": base64.b64encode(PNG).decode()}
        return super().__call__(method, params)


class FluxGui(OnetoneGui):
    def __init__(self, snapshot=None):
        self.snapshot = snapshot
        super().__init__(
            {
                "$resonator.freq": 6100.0,
                "$resonator.kappa": 4.0,
                "$qubit.flux_half": 0.001,
                "$qubit.flux_int": 0.003,
            },
            experiment="onetone/flux_dep",
        )
        root = self.publication["tree"]["children"]
        root["dev"] = section(flux_dev=scalar("flux_yoko"))
        root["flux_unit"] = scalar("A")
        root["sweep"]["children"]["flux"] = {
            "kind": "sweep",
            "valid": True,
            "inputs": {
                "start": self.input("2 * $qubit.flux_int - $qubit.flux_half"),
                "stop": self.input("2 * $qubit.flux_half - $qubit.flux_int"),
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
                "snapshot": self.snapshot
                if self.snapshot is not None
                else {
                    "name": params["name"],
                    "unit": "A" if params["name"] == "coil" else "V",
                }
            }
        return super().__call__(method, params)
