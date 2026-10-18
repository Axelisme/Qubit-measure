"""Fixed-capture roles and editable calibration inputs for joint readout."""

from zcu_tools.cfg_model import Expr, Locked, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.components.mux import MuxGroupSchema
from zcu_lab.v2._support.measure.mux_bindings import validate_mux_bindings

from .core import JointReadoutCfg


def make_form(ctx: FormContext) -> Form[JointReadoutCfg]:
    """Require captured F1/F2 fluxoniums, transmon coupler and full mux binding.

    Resolve roles and original tone positions once at Open/Reset, locking them
    against manual editing. The shared validator owns qubit/resonator relations.
    Missing capture, roles, required kinds or group membership raise ValueError;
    source and alias failures propagate. Never refresh or mutate the capture.
    Xπ/readout slots are copied into editable inputs. Missing pulses or GE
    centers remain Invalid; absent readout starts blank for manual calibration.
    Core validation requires its tone count to match the captured full group.
    """
    validate_mux_bindings(ctx.parameters)
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("Joint readout requires a fixed parameter capture")
    components = parameters.document().components
    f1, f2, coupler, mux = (
        parameters.resolve_path(f"${role}") for role in ("f1", "f2", "coupler", "mux")
    )
    for role, name, kind in (
        ("f1", f1, "qubit/fluxonium"),
        ("f2", f2, "qubit/fluxonium"),
        ("coupler", coupler, "qubit/transmon"),
    ):
        component = components.get(name)
        if component is None or component.kind != kind:
            raise ValueError(f"${role}={name!r} must name a {kind} component")
    if f1 == f2:
        raise ValueError("F1 and F2 must be distinct fluxonium components")
    group = components.get(mux)
    if not isinstance(group, MuxGroupSchema) or group.qubits is None:
        raise ValueError("Joint readout requires a MuxGroup with ordered qubits")
    if f1 not in group.qubits or f2 not in group.qubits:
        raise ValueError("Captured $mux.qubits must contain both F1 and F2")
    return Form(
        JointReadoutCfg,
        tree={
            "roles": {
                "f1": Locked(f1),
                "f2": Locked(f2),
                "coupler": Locked(coupler),
                "mux": Locked(mux),
                "f1_tone": Locked(group.qubits.index(f1)),
                "f2_tone": Locked(group.qubits.index(f2)),
                "tone_count": Locked(len(group.qubits)),
            },
            "modules": {
                "x180_f1": Slot("$f1.module.x180", copy=True),
                "x180_f2": Slot("$f2.module.x180", copy=True),
                "readout": Slot("$mux.module.readout", copy=True)
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
