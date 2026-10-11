"""Shared cfg field recording collaborator for generic and domain owners."""

from ._recipe_support import LookbackGui, scalar, section


class SweepGui(LookbackGui):
    """Cfg collaborator that resolves declared expressions like the GUI owner."""

    def __init__(self):
        super().__init__()
        self.md.update(
            {
                "$resonator.freq": 6100.0,
                "$resonator.kappa": 4.0,
                "$qubit.freq": 4500.0,
                "$qubit.kappa": 3.0,
                "$qubit.flux_half": 0.001,
                "$qubit.flux_int": 0.003,
            }
        )
        self.device = {
            "unit": "A",
            "type_name": "YOKOGS200",
            "info": {"type": "YOKOGS200"},
        }
        self.publication["tree"]["children"].update(
            dev=section(flux_dev=scalar("coil")),
            flux_unit=scalar("A"),
            sweep=section(
                freq={
                    "kind": "sweep",
                    "valid": True,
                    "readonly": False,
                    "inputs": {
                        "start": self.input("$resonator.freq - 2 * $resonator.kappa"),
                        "stop": self.input("$resonator.freq + 3 * $resonator.kappa"),
                        "expts": self.input(41),
                    },
                },
                flux={
                    "kind": "sweep",
                    "valid": True,
                    "readonly": False,
                    "inputs": {
                        "start": self.input("2 * $qubit.flux_int - $qubit.flux_half"),
                        "stop": self.input("2 * $qubit.flux_half - $qubit.flux_int"),
                        "expts": self.input(19),
                    },
                },
            ),
        )

    def input(self, value: object) -> dict[str, object]:
        """Resolve a scalar or expression against this collaborator's md."""
        result: dict[str, object] = scalar(
            self.evaluate_expression(value) if isinstance(value, str) else value
        )["input"]
        if isinstance(value, str):
            result.update(mode="expression", raw=value)
        return result

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
            return {"values": [{"key": "device.flux.name"}]}
        if method == "value.read":
            assert params == {"key": "device.flux.name"}
            return {"value": "coil"}
        if method == "device.snapshot":
            return {"snapshot": {"name": params["name"], **self.device}}
        return super().__call__(method, params)
