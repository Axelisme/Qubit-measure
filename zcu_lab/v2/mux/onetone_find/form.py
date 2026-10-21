"""Open/Reset defaults for a single-tone absolute mux frequency survey."""

from zcu_tools.cfg_model import Hidden, Range
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_bindings import validate_mux_bindings
from zcu_lab.v2._support.measure.mux_forms import READOUT_ELEMENTS, mux_readout_initial

from .core import MuxOnetoneFindCfg


def make_form(ctx: FormContext) -> Form[MuxOnetoneFindCfg]:
    """Copy the captured mux readout and generate editable group tones.

    ctx supplies the fixed Open/Reset capture. Reuse the fine-scan helper to
    copy scalar readout settings from $mux.module.readout and regenerate tones
    from the group resonators. Missing readout slots use the blank mux template.
    The same helper adds five ringup times using the smallest set group kappa;
    unset kappas add no wait and invalid set kappas raise. Sources stay unchanged.
    Invalid declared qubit/resonator pairings also reject Open/Reset.

    tone starts at original index 0. Absolute MHz start/stop are required operator
    input, with 501 points initially. Run rejects values outside the generator
    mux band; opening this form does not authorize hardware acquisition.
    """
    validate_mux_bindings(ctx.parameters)
    return Form(
        MuxOnetoneFindCfg,
        tree={
            "modules": {
                "readout": mux_readout_initial(ctx, wait_for_ringup=True),
            },
            "sweep": {"freq": Range.between(None, None, expts=501)},
            "initial_delay": Hidden(),
        },
        fixed_elements=(READOUT_ELEMENTS,),
        field_order=("tone", "sweep", "modules", "relax_delay", "reps", "rounds"),
    )
