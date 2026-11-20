"""Fixed-capture calibration and editable inputs for CZ optimization."""

from zcu_tools.cfg_model import Expr, Slot
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata

from .._measure import captured_roles
from .core import CZOptimizeCfg


def make_form(ctx: FormContext) -> Form[CZOptimizeCfg]:
    """Capture calibration slots and ordered CZ corrections at Open or Reset.

    Missing sources remain invalid, and the optional reset starts unset.
    Operators fill the fixed depth, sequence seeds and CMA settings before Run.
    """
    names: dict[str, str] = {}
    if ctx.parameters is not None:
        for role in ("f1", "f2"):
            path = f"$coupler.ext.cz_phase.{role}"
            captured = ctx.parameters.lookup(path)
            if captured.present:
                if not isinstance(captured.value, str):
                    raise ValueError(f"{path} must contain a component name")
                names[role] = captured.value
    readout = Slot("$mux.module.readout")
    return Form(
        CZOptimizeCfg,
        tree={
            "reps": 1000,
            "relax_delay": 1.0,
            "roles": captured_roles(ctx),
            "reset": None,
            "x90_f1": Slot("$f1.module.x90", copy=True),
            "x90_f2": Slot("$f2.module.x90", copy=True),
            "x180_f1": Slot("$f1.module.x180", copy=True),
            "x180_f2": Slot("$f2.module.x180", copy=True),
            "drive": {
                "f1": Slot("$coupler.module.cz_f1", copy=True),
                "f2": Slot("$coupler.module.cz_f2", copy=True),
            },
            "readout": readout,
            "phase_pair": {
                **names,
                "correction_f1_deg": Expr("$coupler.ext.cz_phase.correction_f1_deg"),
                "correction_f2_deg": Expr("$coupler.ext.cz_phase.correction_f2_deg"),
            },
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
