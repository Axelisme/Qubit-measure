"""Fixed-capture roles and editable calibration inputs for joint readout."""

from zcu_tools.cfg_model import Expr, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_forms import mux_readout_seed_metadata

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
    readout = (
        Slot("$mux.module.readout")
        if ctx.has("$mux.module.readout")
        else Template("readout/mux:blank")
    )
    return Form(
        JointReadoutCfg,
        field_order=(
            "roles",
            "modules",
            "f1_centers",
            "f2_centers",
            "reps",
            "rounds",
            "relax_delay",
            "initial_delay",
        ),
        tree={
            "reps": 1000,
            "rounds": 10,
            "roles": roles,
            "modules": {
                "x180_f1": Slot("$f1.module.x180", copy=True),
                "x180_f2": Slot("$f2.module.x180", copy=True),
                "readout": readout,
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
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
