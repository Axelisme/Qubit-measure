"""Mux readout defaults from the fixed Open/Reset capture."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.mux.onetone.core import MuxOnetoneCfg


def make_form(ctx: FormContext) -> Form[MuxOnetoneCfg]:
    """Use the captured mux slot, or a blank editable readout without tones.

    Missing group bindings do not gate Run: valid manual readout inputs suffice.
    Active tones defaults to None (all), as declared by the core cfg.
    """
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
