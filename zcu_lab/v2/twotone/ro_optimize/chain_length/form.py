"""Starting values for ro_optimize/chain_length; Qt-free data only."""

from collections.abc import Mapping
from dataclasses import replace

from zcu_tools.cfg_model import FormNode, Hidden, Range, Template, TemplateDef
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.readout_initial_values import (
    pulse_readout_length_margin,
)
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata
from zcu_lab.v2.twotone.ro_optimize.chain_length.core import ChainLengthCfg


def make_form(
    ctx: FormContext, *, templates: Mapping[str, TemplateDef]
) -> Form[ChainLengthCfg]:
    """Seed the editable length margin in us from the captured readout or template.

    Use templates from the registering Registry only when ctx has no captured
    readout. An unknown template ID raises ValueError. Leave the margin required
    and blank if either length cannot be resolved. Seed thermal relaxation K from
    5*T1 using the same fixed Open/Reset context.
    """
    readout = source(
        ctx,
        "$resonator.module.readout",
        Template(
            "readout",
            bindings={"resonator": "$resonator"},
            seed_metadata=pulse_readout_seed_metadata(gain=0.1),
        ),
    )
    margin = pulse_readout_length_margin(ctx, readout, templates=templates)
    # Run replaces both lengths for each point in the ADC window sweep.
    readout = replace(
        readout,
        set={
            **readout.set,
            "ro_length": Hidden(0.0),
            "waveform.length": Hidden(0.0),
        },
    )
    tree: dict[str, FormNode] = {
        "reps": 2000,
        "rounds": 320,
        "readout": readout,
        "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
        "length": Range(0.2, 2.2, expts=31),
    }
    if margin.value is not None:
        tree["length_margin"] = margin.value
    return Form(ChainLengthCfg, tree).with_seed_metadata(
        {("length_margin",): margin.metadata}
    )
