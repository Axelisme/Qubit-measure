"""Real module captures shared by the public plan and Qt attachment cases."""

from zcu_tools.cfg_model import PulseReadoutSpec, PulseSpec
from zcu_tools.gui.form import FormEnv, FormInput
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)

from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.workflows.gui.plan import T1PlanForm


def inline_input(inputs: Inputs) -> FormInput[T1PlanForm]:
    """Select the simulator drive/readout as ordinary inline module Specs."""
    return FormInput(
        T1PlanForm,
        {
            "flux_dev": "flux",
            "fluxes": {"kind": "list", "values": [1.0, 1.1]},
            "points": 21,
            "qub_pulse": {
                "kind": "inline",
                "module": inputs.rabi.modules.qub_pulse.model_dump(mode="json"),
            },
            "readout": {
                "kind": "inline",
                "module": inputs.readout.modules.readout.model_dump(mode="json"),
            },
        },
    )


def library_env(inputs: Inputs) -> FormEnv:
    """Capture actual library roots for the same two native module selections."""
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="1.0",
                components={
                    "Q1": ComponentLibrary(
                        pulses={
                            "pi": PulseSpec.model_validate(
                                inputs.rabi.modules.qub_pulse.model_dump(mode="python")
                            )
                        },
                        readout={
                            "ro": PulseReadoutSpec.model_validate(
                                inputs.readout.modules.readout.model_dump(mode="python")
                            )
                        },
                    )
                },
            )
        )
    )
