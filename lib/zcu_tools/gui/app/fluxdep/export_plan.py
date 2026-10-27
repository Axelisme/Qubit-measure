"""Injected component calibration policy for the standalone fluxdep exporter."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

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
    integer/half positions. It must reject invalid calibration before I/O, and
    must not infer missing device identities or contemporaneous instrument values.
    """

    fields: Mapping[str, float]
    flux_writeback: FluxWritebackReceipt


ExportPlanner = Callable[[ComponentSchema, AlignmentPublication], ComponentExportPlan]
