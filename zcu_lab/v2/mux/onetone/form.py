"""Mux readout defaults from the fixed Open/Reset capture."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_bindings import validate_mux_bindings
from zcu_lab.v2.mux.onetone.core import MuxOnetoneCfg


def make_form(ctx: FormContext) -> Form[MuxOnetoneCfg]:
    """Use the captured mux slot, or a blank editable readout without tones.

    With qubits, reject missing/wrong-kind components or mismatched resonators
    from the fixed capture at Open/Reset. Group structure is model-validated.
    Without qubits, missing bindings do not gate Run: manual readout suffices.
    Active tones defaults to None (all), as declared by the core cfg.
    Source and alias faults propagate; no live files or devices are read.
    """
    validate_mux_bindings(ctx.parameters)
    return Form(
        MuxOnetoneCfg,
        tree={
            "modules": {
                "reset": Hidden(),
                "readout": Slot("$mux.module.readout")
                if ctx.has("$mux.module.readout")
                else Template("readout/mux:blank"),
            },
            "sweep": {"df": Range.between(-1.0, 1.0, expts=101)},
            "initial_delay": Hidden(),
        },
    )
