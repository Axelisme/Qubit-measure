"""Component calibration policy is injected at the public export seam."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
from zcu_tools.gui.app.fluxdep.export_plan import (
    ComponentExportPlan,
    FluxWritebackReceipt,
)
from zcu_tools.gui.app.fluxdep.services.fit import FitService
from zcu_tools.gui.app.fluxdep.state import AlignmentPublication
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.resources.entry import ResultEntry, component_registry, snapshot
from zcu_tools.resources.entry.schema import ComponentSchema
from zcu_tools.resources.predictor_file import load_predictor_file

from tests.gui.app.fluxdep._components import skip_calibration
from tests.gui.app.fluxdep._export import aligned_export_state


class CalibratedComponent(ComponentSchema):
    """Synthetic export target with a lab-defined scalar calibration."""

    kind: str = "qubit/fluxonium/export"
    calibration: float = 0.0
    EJ: float | None = None
    EC: float | None = None
    EL: float | None = None


@pytest.fixture
def export_target(fluxonium_entry: ResultEntry) -> Iterator[AnalysisTarget]:
    component_registry.register("qubit/fluxonium/export", CalibratedComponent)
    try:
        fluxonium_entry.setup.add_component("Qcal", kind="qubit/fluxonium/export")
        yield AnalysisTarget.open(
            fluxonium_entry.result_path.parent.parent, "synthetic", "Qcal"
        )
    finally:
        component_registry.unregister("qubit/fluxonium/export")


def test_export_applies_injected_calibration_and_retains_predictor_and_ledger(
    export_target: AnalysisTarget,
) -> None:
    target = export_target
    state = aligned_export_state(target)
    calls: list[tuple[str, AlignmentPublication]] = []

    def plan(
        model: ComponentSchema, points: AlignmentPublication
    ) -> ComponentExportPlan:
        calls.append((model.kind, points))
        return ComponentExportPlan(
            {"calibration": points.flux_int},
            FluxWritebackReceipt("written", "synthetic-bias", None),
        )

    service = FitService(state)
    service.configure_export(plan)
    receipt = service.export_params_receipt()
    target.resource.setup.refresh()
    assert (
        snapshot(target.resource.setup).components["Qcal"].model_dump()["calibration"]
        == 1.7
    )
    assert calls == [
        (
            "qubit/fluxonium/export",
            AlignmentPublication(flux_int=1.7, flux_half=0.7, flux_period=2.0),
        )
    ]
    assert receipt.flux_writeback == FluxWritebackReceipt(
        "written", "synthetic-bias", None
    )
    predictor = load_predictor_file(Path(receipt.savepath))
    assert predictor.params.flux_half == 0.7
    assert predictor.params.flux_period == 2.0
    assert receipt.params == (5.0, 1.2, 0.4)
    assert target.resource.ledger.get(receipt.analysis_id).event.kind == "analyzed"


def test_export_skip_keeps_calibration_and_writes_energies_and_predictor(
    export_target: AnalysisTarget,
) -> None:
    service = FitService(aligned_export_state(export_target))
    service.configure_export(skip_calibration)
    receipt = service.export_params_receipt()
    model = snapshot(export_target.resource.setup).components["Qcal"].model_dump()
    assert model["calibration"] == 0.0
    assert (model["EJ"], model["EC"], model["EL"]) == (5.0, 1.2, 0.4)
    assert receipt.flux_writeback.status == "skipped"
    assert receipt.flux_writeback.reason is not None
    assert Path(receipt.savepath).is_file()
    assert (
        export_target.resource.ledger.get(receipt.analysis_id).event.kind == "analyzed"
    )


@pytest.mark.parametrize("missing_planner", [False, True])
def test_export_policy_failure_precedes_output(
    export_target: AnalysisTarget, missing_planner: bool
) -> None:
    def reject(
        _model: ComponentSchema, _points: AlignmentPublication
    ) -> ComponentExportPlan:
        raise ValueError("invalid synthetic calibration")

    service = FitService(aligned_export_state(export_target))
    if not missing_planner:
        service.configure_export(reject)
    expected = FailedPreconditionError if missing_planner else ValueError
    diagnostic = "planner is not configured" if missing_planner else "invalid synthetic"
    with pytest.raises(expected, match=diagnostic):
        service.export_params_receipt()
    assert not export_target.predictor_path.exists()
    assert (
        snapshot(export_target.resource.setup).components["Qcal"].model_dump()["EJ"]
        is None
    )
