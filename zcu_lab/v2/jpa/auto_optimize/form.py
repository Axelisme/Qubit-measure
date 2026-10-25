"""Starting values for jpa/auto_optimize; Qt-free declarations."""

from zcu_tools.cfg_model import Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.jpa.auto_optimize.core import JPAOptCfg
from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time


def make_form(ctx: FormContext) -> Form[JPAOptCfg]:
    """Build starting values and shape from the captured source availability."""
    flux_source = (
        ctx.parameters.lookup_point("$jpa.flux_source").value
        if ctx.parameters is not None
        else None
    )
    if flux_source is not None and not isinstance(flux_source, str):
        raise ValueError("$jpa.flux_source must name a device")
    rf_source = (
        ctx.parameters.lookup_point("$jpa.rf_source").value
        if ctx.parameters is not None
        else None
    )
    if rf_source is not None and not isinstance(rf_source, str):
        raise ValueError("$jpa.rf_source must name a device")
    return Form(
        JPAOptCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180")
            if ctx.has("$qubit.module.x180")
            else Template("pi_pulse", bindings={"qubit": "$qubit"}),
            "readout": Slot("$resonator.module.readout")
            if ctx.has("$resonator.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "jpa_rf_dev": rf_source,
            "jpa_flux_dev": flux_source,
            "flux_unit": "A",
            "jpa_flux": Range.around("$jpa.flux", 0.005, expts=101),
            "jpa_freq": Range(
                "0.98 * $jpa.pump_freq", "1.02 * $jpa.pump_freq", expts=101
            ),
            "jpa_power": Range.around("$jpa.pump_power", 5.0, expts=101),
            "num_points": 1001,
            "skew_penalty": 0.0,
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=1),
            "reps": 1000,
            "rounds": 10,
        },
    )
