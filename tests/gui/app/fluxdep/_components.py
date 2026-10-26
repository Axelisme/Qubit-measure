"""Synthetic component schemas shared by fluxdep analysis tests."""

from zcu_tools.analysis.flux import DeviceFluxPoint
from zcu_tools.gui.app.fluxdep.export_plan import (
    ComponentExportPlan,
    FluxWritebackReceipt,
)
from zcu_tools.resources.entry.schema import ComponentSchema


class SyntheticFluxonium(ComponentSchema):
    """Analysis target declaring kind and EJ/EC/EL energies in GHz, but no flux fields."""

    kind: str = "qubit/fluxonium"
    EJ: float | None = None
    EC: float | None = None
    EL: float | None = None


def skip_calibration(
    _model: ComponentSchema, _points: DeviceFluxPoint
) -> ComponentExportPlan:
    """Synthetic policy with no calibration fields; explicitly report the omission."""
    return ComponentExportPlan(
        {},
        FluxWritebackReceipt(
            "skipped",
            None,
            "Synthetic target has no calibration. Use fluxdep measurement writeback.",
        ),
    )
