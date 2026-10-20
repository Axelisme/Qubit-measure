"""Fixed-capture roles and editable calibration inputs for joint readout."""

from zcu_tools.cfg_model import Expr, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from .._measure import captured_roles
from .core import JointReadoutCfg


def make_form(ctx: FormContext) -> Form[JointReadoutCfg]:
    """Require captured F1/F2 fluxoniums, transmon coupler and full mux binding.

    Resolve roles and original tone positions once at Open/Reset, locking them
    against manual editing. The shared validator owns qubit/resonator relations.
    Missing capture, roles, required kinds or group membership raise ValueError;
    source and alias failures propagate. Never refresh or mutate the capture.
    Xπ slots are copied into editable inputs; readout references its library slot.
    Missing pulses or GE centers remain Invalid; absent readout starts blank
    for manual calibration.
    Core validation requires its tone count to match the captured full group.
    """
    roles = captured_roles(ctx)
    return Form(
        JointReadoutCfg,
        tree={
            "roles": roles,
            "modules": {
                "x180_f1": Slot("$f1.module.x180", copy=True),
                "x180_f2": Slot("$f2.module.x180", copy=True),
                "readout": Slot("$mux.module.readout")
                if ctx.has("$mux.module.readout")
                else Template("readout/mux:blank"),
            },
            "f1_centers": {
                "g": Expr("$f1.ext.readout_cal.g_center"),
                "e": Expr("$f1.ext.readout_cal.e_center"),
            },
            "f2_centers": {
                "g": Expr("$f2.ext.readout_cal.g_center"),
                "e": Expr("$f2.ext.readout_cal.e_center"),
            },
        },
    )
