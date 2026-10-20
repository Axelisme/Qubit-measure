"""Fixed capture FTF conditional Ramsey form, with editable calibrated pulses."""

from zcu_tools.cfg_model import Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from .._measure import captured_roles
from .core import IdleZZCfg


def make_form(ctx: FormContext) -> Form[IdleZZCfg]:
    """Lock captured role/tone mapping, seed pulses/readout without model truth.

    Require the same complete FTF roles/group as joint readout, delegating mux
    relationships to the public validator before building the Form. Missing Xπ
    or Xπ/2 remains Invalid. Readout may be manually calibrated if no slot exists.
    Optional DC source starts disabled; enabling it requires a direct connected device
    name, working unit and authorized source-coordinate sweep. Open/Reset is the
    only time sources are read; Run and historical analysis never consult it.
    """
    roles = captured_roles(ctx)
    return Form(
        IdleZZCfg,
        tree={
            "roles": roles,
            "modules": {
                "x180_f1": Slot("$f1.module.x180", copy=True),
                "x180_f2": Slot("$f2.module.x180", copy=True),
                "x90_f1": Slot("$f1.module.x90", copy=True),
                "x90_f2": Slot("$f2.module.x90", copy=True),
                "readout": Slot("$mux.module.readout", copy=True)
                if ctx.has("$mux.module.readout")
                else Template("readout/mux:blank"),
            },
            "detune": None,
            "delay": Range.between(0, 10, expts=101),
            "dc_device": None,
            "dc_values": Range.between(0, 0, expts=1),
        },
    )
