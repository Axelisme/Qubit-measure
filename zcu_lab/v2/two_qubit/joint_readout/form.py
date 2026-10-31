"""Fixed-capture roles and editable calibration inputs for joint readout."""

from typing import TypeVar

from zcu_tools.cfg_model import Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata

from .._measure import captured_roles
from .core import JointReadoutCfg
from .core_heralded import JointReadoutHeraldedCfg

T_Cfg = TypeVar("T_Cfg", JointReadoutCfg, JointReadoutHeraldedCfg)


def joint_readout_form(ctx: FormContext, cfg_type: type[T_Cfg]) -> Form[T_Cfg]:
    """Require captured F1/F2 fluxoniums, transmon coupler and full mux binding.

    Resolve roles and original tone positions once at Open/Reset, locking them
    against manual editing. The shared validator owns qubit/resonator relations.
    Missing capture, roles, required kinds or group membership raise ValueError;
    source and alias failures propagate. Never refresh or mutate the capture.
    Xπ slots are copied into editable inputs; readout references its library slot.
    Missing pulses remain Invalid; absent readout starts blank
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
        cfg_type,
        field_order=(
            "roles",
            "reset",
            "x180_f1",
            "x180_f2",
            "readout",
            "reps",
            "rounds",
            "relax_delay",
            "initial_delay",
        ),
        tree={
            "reps": 1000,
            "rounds": 10,
            "roles": roles,
            "reset": None,
            "x180_f1": Slot("$f1.module.x180", copy=True),
            "x180_f2": Slot("$f2.module.x180", copy=True),
            "readout": readout,
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))


def make_form(ctx: FormContext) -> Form[JointReadoutCfg]:
    """Capture full mux roles and seed editable pulses/readout; reset starts None."""
    return joint_readout_form(ctx, JointReadoutCfg)
