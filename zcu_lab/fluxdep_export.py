"""Lab component calibration policy injected into the standalone fluxdep app."""

from zcu_tools.analysis.flux import DeviceFluxPoint
from zcu_tools.cfg_model import validate_device_name
from zcu_tools.gui.app.fluxdep.export_plan import (
    ComponentExportPlan,
    FluxWritebackReceipt,
)
from zcu_tools.resources.entry.schema import ComponentSchema

from zcu_lab.components.fluxonium import QubitSchema
from zcu_lab.components.qubit import QubitFluxCalibration
from zcu_lab.v2.support.flux import qubit_flux_frame, reference_device_flux_points


def plan_fluxdep_export(
    component: ComponentSchema, points: DeviceFluxPoint
) -> ComponentExportPlan:
    """Plan setup flux calibration only for a single-device fluxonium frame.

    component is a detached lab fluxonium projection. points holds fitted native
    integer/half positions from the standalone app. Only a frame containing
    exactly flux_dev is writable without other-device values. Preserve its
    period sign and update the magnitude from the fitted separation. Return
    component-relative scalar paths, without modifying the projection or files.

    Missing source, empty/mismatched frame or multiple contributors return a
    skipped receipt directing the operator to measurement writeback. Unknown
    component types, invalid names or inconsistent stored calibration raise
    ValueError before export. A writable frame also rejects zero fitted span.
    No device catalog or hardware is consulted.
    """
    if not isinstance(component, QubitSchema):
        raise ValueError("fluxdep export requires a lab fluxonium component")
    qubit_flux_frame(component.flux)
    source = component.flux_dev
    if source is not None:
        validate_device_name(source)
    if source is None:
        reason = "flux_dev is not configured."
    elif not component.flux:
        reason = "The component flux frame is empty."
    elif set(component.flux) != {source}:
        reason = (
            "The flux frame is not limited to flux_dev; standalone alignment "
            "does not identify the swept device or other devices' values at acquisition."
        )
    else:
        fitted = QubitFluxCalibration(
            half=points.half,
            integer=points.integer,
            period=2.0 * abs(points.integer - points.half),
        )
        reference = reference_device_flux_points(component.flux, source, fitted, {})
        return ComponentExportPlan(
            {
                f"flux.{source}.half": reference.half,
                f"flux.{source}.integer": reference.integer,
                f"flux.{source}.period": reference.period,
            },
            FluxWritebackReceipt("written", source, None),
        )
    return ComponentExportPlan(
        {},
        FluxWritebackReceipt(
            "skipped",
            source,
            f"{reason} Use fluxdep measurement writeback to calibrate the component.",
        ),
    )
