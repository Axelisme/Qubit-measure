"""Real Form recording fixture for two independent authored ranges."""

from zcu_tools.cfg_model import ConfigBase, Range, Slot, Template
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2 import SweepCfg

from ._recipe_support import READOUT_TEMPLATE, LookbackGui, RecipeControlsBase


class SweepFields(ConfigBase):
    freq: SweepCfg
    flux: SweepCfg


class SweepControls(RecipeControlsBase):
    sweep: SweepFields


class SweepGui(LookbackGui):
    """Use public source capture, expression evaluation and range leaf writes."""

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

        form = Form(
            SweepControls,
            {
                "modules": {
                    "reset": Slot("$qubit.module.reset"),
                    "init_pulse": Slot("$qubit.module.init_pulse"),
                    "readout": Template("readout"),
                },
                "sweep": {
                    "freq": Range.between(
                        "$resonator.freq - 2 * $resonator.kappa",
                        "$resonator.freq + 3 * $resonator.kappa",
                        expts=41,
                    ),
                    "flux": Range.between(
                        "2 * $qubit.flux_int - $qubit.flux_half",
                        "2 * $qubit.flux_half - $qubit.flux_int",
                        expts=19,
                    ),
                },
            },
        )
        form.bind_templates({"readout": READOUT_TEMPLATE}, adapter="lookback")

        def capture(aliases):
            parameters = self.capture_parameters(aliases)
            return TabFormSources(
                FormEnv(parameters=parameters, library=parameters.library)
            )

        self.owner = TabFormResource(
            lambda _ctx: form, capture=capture, mutation_allowed=lambda: True
        )

    def __call__(self, method, params):
        if method == "value.list":
            return {"values": [{"key": "device.flux.name"}]}
        if method == "value.read":
            assert params == {"key": "device.flux.name"}
            return {"value": "coil"}
        if method == "device.snapshot":
            return {"snapshot": {"name": params["name"], **self.device}}
        return super().__call__(method, params)
