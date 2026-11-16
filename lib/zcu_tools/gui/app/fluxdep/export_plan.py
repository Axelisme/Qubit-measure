"""Injected component calibration policy for the standalone fluxdep exporter."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from zcu_tools.analysis.flux import require_matching_device_unit
from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
from zcu_tools.gui.app.analysis_project.flux import (
    ComponentFluxProvider,
    component_flux_context,
)
from zcu_tools.resources.entry import snapshot
from zcu_tools.resources.entry.schema import ComponentSchema

from .state import AlignmentPublication


@dataclass(frozen=True)
class FluxWritebackReceipt:
    """Independent component or dataset export outcome shared by GUI/RPC/MCP.

    status is 'written' when the component plan supplies calibration fields or
    the dataset predictor is written, otherwise 'skipped'. Each outcome has its
    own receipt. device names the calibrated or scanned instrument, or the
    intended instrument when known for 'skipped'. reason is None for 'written';
    'skipped' requires a nonblank explanation and the next operator action.
    Invalid status/field combinations raise ValueError at construction.
    """

    status: Literal["written", "skipped"]
    device: str | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.status == "written":
            if (
                self.device is None
                or not self.device.strip()
                or self.reason is not None
            ):
                raise ValueError("written requires a device and no skip reason")
        elif self.status != "skipped" or self.reason is None or not self.reason.strip():
            raise ValueError("skipped requires a reason and next operator action")


@dataclass(frozen=True)
class ComponentExportPlan:
    """Calibration writes and outcome produced without I/O by the lab planner.

    fields maps component-relative dotted scalar paths to native numeric values.
    It excludes the fit's EJ/EC/EL, which the exporter writes when declared.
    flux_writeback explains whether calibration fields are supplied or skipped.
    The planner receives a detached ComponentSchema projection and fitted native
    integer/half positions. Alignment and processed restore may query it before
    fitting to validate the written receipt's target unit. Planning must not
    perform I/O writes or require fit results. It must reject invalid calibration
    before I/O, and must not infer missing device identities or contemporaneous
    instrument values.
    """

    fields: Mapping[str, float]
    flux_writeback: FluxWritebackReceipt


ExportPlanner = Callable[[ComponentSchema, AlignmentPublication], ComponentExportPlan]


def require_alignment_export_unit(
    target: AnalysisTarget | None,
    planner: ExportPlanner | None,
    provider: ComponentFluxProvider | None,
    alignment: AlignmentPublication,
    unit: str,
) -> None:
    """Reject an axis unit that cannot calibrate the planner's actual device.

    Query the injected planner with detached global and real native alignment,
    without writing its fields. Only a written receipt selects a device for
    exact context-unit comparison; absent context units mean native. No target,
    no planner or a skipped receipt leaves alignment unrestricted. Planner and
    provider errors propagate; unit mismatch raises ValueError naming the device
    and both units. No fit, I/O writes or unit conversion are performed.
    """
    if target is None or planner is None:
        return
    model = snapshot(target.resource.parameters()).components[target.component]
    receipt = planner(model, alignment).flux_writeback
    if receipt.status == "written" and receipt.device is not None:
        context = component_flux_context(target, provider)
        require_matching_device_unit(
            receipt.device, unit, context.units.get(receipt.device)
        )
