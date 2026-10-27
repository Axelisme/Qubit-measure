"""Lab period calibration policy injected into the standalone fluxdep app."""

from math import copysign

from zcu_tools.cfg_model import validate_device_name
from zcu_tools.gui.app.fluxdep.export_plan import (
    ComponentExportPlan,
    FluxWritebackReceipt,
)
from zcu_tools.gui.app.fluxdep.state import AlignmentPublication
from zcu_tools.resources.entry.schema import ComponentSchema

from zcu_lab.components.fluxonium import QubitSchema
from zcu_lab.components.qubit import QubitFluxCalibration


def plan_fluxdep_export(
    component: ComponentSchema, points: AlignmentPublication
) -> ComponentExportPlan:
    """Update the default flux device's signed period, never its component offset.

    component is a detached lab fluxonium projection. points holds the app's
    fitted native integer/half positions. Preserve the stored period direction,
    initially positive, and update its magnitude. Other device periods and the
    current offset are unchanged: dataset alignment can predate a flux jump.
    The exporter separately writes declared EJ/EC/EL fields.

    Missing flux_dev returns a skipped receipt with the next operator action.
    Wrong component types, invalid names, nonfinite points or zero fitted span
    raise ValueError before export. No snapshot, device catalog or I/O is used.
    """
    if not isinstance(component, QubitSchema):
        raise ValueError("fluxdep export requires a lab fluxonium component")
    source = component.flux_dev
    if source is None:
        return ComponentExportPlan(
            {},
            FluxWritebackReceipt(
                "skipped",
                None,
                "flux_dev is not configured. Set the swept device name before exporting its period.",
            ),
        )
    validate_device_name(source)
    fitted = QubitFluxCalibration(
        half=points.flux_half,
        integer=points.flux_int,
        period=2.0 * abs(points.flux_int - points.flux_half),
    )
    period = copysign(abs(fitted.period), component.flux.period.get(source, 1.0))
    return ComponentExportPlan(
        {f"flux.period.{source}": period},
        FluxWritebackReceipt("written", source, None),
    )
