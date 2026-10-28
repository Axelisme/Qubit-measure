"""Component calibration policy is injected at the public export seam."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from zcu_tools.analysis.flux import ComponentFluxContext, DevicePeriod, FluxFrame
from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
from zcu_tools.gui.app.analysis_project.selection import AnalysisSelection
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

from tests.gui.app.fluxdep._components import skip_calibration, synthetic_flux_context
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
            AnalysisSelection(
                fluxonium_entry.result_path.parent.parent, "synthetic", "Qcal"
            )
        )
    finally:
        component_registry.unregister("qubit/fluxonium/export")


@pytest.mark.parametrize("other_unit", ["V", "native"])
@pytest.mark.parametrize("snapshot_kind", ["present", "missing_other", "empty"])
def test_export_v2_uses_fitted_swept_period_and_other_snapshot_contribution(
    export_target: AnalysisTarget,
    snapshot_kind: str,
    other_unit: str,
) -> None:
    state = aligned_export_state(export_target)
    entry = state.get_spectrum("aligned")
    entry.device = "swept"
    # Component period is skipped, so the dataset may use a different axis unit.
    entry.unit = "mA"
    entry.device_snapshot = {
        # Swept and out-of-frame snapshot units are not used by the inverse.
        "swept": {"value": 99.0, "unit": "V"},
        "unrelated": {"value": 2.0, "unit": "Hz"},
    }
    if snapshot_kind == "present":
        entry.device_snapshot["coil"] = {"value": 1.0, "unit": other_unit}
    elif snapshot_kind == "empty":
        entry.device_snapshot = {}
    context = ComponentFluxContext(
        FluxFrame(
            0.9,
            {"coil": DevicePeriod(-4.0, other_unit), "swept": DevicePeriod(-8.0, "A")},
        ),
        "coil",
        {"swept": "A", "coil": other_unit},
    )
    service = FitService(state)
    service.configure_export(skip_calibration)
    service.configure_flux_context(lambda _model: context)

    receipt = service.export_params_receipt()

    assert receipt.predictor_writeback == FluxWritebackReceipt("written", "swept", None)
    assert receipt.savepath is not None
    predictor = load_predictor_file(Path(receipt.savepath))
    assert list(predictor.flux.period) == ["swept", "coil"]
    assert predictor.params.to_dict() == {"EJ": 5.0, "EC": 1.2, "EL": 0.4}
    assert predictor.flux.to_dict() == {
        "offset": pytest.approx(0.4 if snapshot_kind == "present" else 0.15),
        "period": {"swept": 2.0, "coil": -4.0},
        "unit": {"swept": "mA", "coil": other_unit},
    }


@pytest.mark.parametrize(
    "case", ["uncalibrated", "missing_snapshot", "unknown", "unnamed"]
)
def test_predictor_skip_retains_component_plan_and_ledger(
    export_target: AnalysisTarget, case: str
) -> None:
    state = aligned_export_state(export_target)
    entry = state.get_spectrum("aligned")
    entry.device = {"unknown": "other", "unnamed": "unnamed_device"}.get(case, "swept")
    entry.unit = "A"
    frame = FluxFrame(
        0.2, {"swept": DevicePeriod(-8.0, "A"), "coil": DevicePeriod(-4.0, "V")}
    )
    if case != "missing_snapshot":
        # No inverse is performed on these skip paths, so this unit is unused.
        entry.device_snapshot = {"coil": {"value": 1.0, "unit": "mA"}}
    context = ComponentFluxContext(
        None if case == "uncalibrated" else frame,
        "swept",
        {name: device.unit for name, device in frame.devices.items()},
    )
    service = FitService(state)
    service.configure_export(
        lambda _model, points: ComponentExportPlan(
            {"calibration": -points.flux_period},
            FluxWritebackReceipt("written", "swept", None),
        )
    )
    service.configure_flux_context(lambda _model: context)

    receipt = service.export_params_receipt()

    assert receipt.savepath is None
    assert receipt.predictor_writeback.status == "skipped"
    assert receipt.predictor_writeback.device == entry.device
    reason = receipt.predictor_writeback.reason
    assert reason is not None
    assert {
        "uncalibrated": "uncalibrated",
        "missing_snapshot": "snapshot",
        "unknown": "other",
        "unnamed": "unnamed_device",
    }[case] in reason
    assert receipt.flux_writeback == FluxWritebackReceipt("written", "swept", None)
    assert not export_target.predictor_path.exists()
    model = snapshot(export_target.resource.setup).components["Qcal"].model_dump()
    assert (model["EJ"], model["EC"], model["EL"], model["calibration"]) == (
        5.0,
        1.2,
        0.4,
        -2.0,
    )
    assert (
        export_target.resource.ledger.get(receipt.analysis_id).event.kind == "analyzed"
    )


def test_predictor_uses_positive_fitted_span_even_for_signed_restored_alignment(
    export_target: AnalysisTarget,
) -> None:
    state = aligned_export_state(export_target)
    entry = state.get_spectrum("aligned")
    entry.flux_period = -2.0
    entry.flux_int = -0.3
    service = FitService(state)
    service.configure_export(skip_calibration)
    service.configure_flux_context(synthetic_flux_context)

    receipt = service.export_params_receipt()

    assert receipt.savepath is not None
    predictor = load_predictor_file(Path(receipt.savepath))
    assert predictor.flux.period == {"unnamed_device": 2.0}
    assert predictor.flux.offset == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("actual_unit", "expected_unit"),
    [("mA", "A"), ("native", "A"), ("A", "native")],
)
def test_used_snapshot_unit_mismatch_fails_before_export(
    export_target: AnalysisTarget, actual_unit: str, expected_unit: str
) -> None:
    state = aligned_export_state(export_target)
    entry = state.get_spectrum("aligned")
    entry.device_snapshot = {"coil": {"value": 1.0, "unit": actual_unit}}
    context = ComponentFluxContext(
        FluxFrame(
            0.2,
            {
                "unnamed_device": DevicePeriod(8.0),
                "coil": DevicePeriod(-4.0, expected_unit),
            },
        ),
        "unnamed_device",
        {"unnamed_device": "native", "coil": expected_unit},
    )
    service = FitService(state)
    service.configure_export(skip_calibration)
    service.configure_flux_context(lambda _model: context)

    with pytest.raises(ValueError, match=f"coil.*{actual_unit}.*{expected_unit}"):
        service.export_params_receipt()

    assert not export_target.predictor_path.exists()
    assert (
        snapshot(export_target.resource.setup).components["Qcal"].model_dump()["EJ"]
        is None
    )
    assert export_target.resource.ledger.read_events_snapshot() == ()


@pytest.mark.parametrize("calibrated", [True, False])
@pytest.mark.parametrize(
    ("axis_unit", "target_unit"),
    [("mA", "A"), ("native", "A"), ("A", None)],
)
def test_component_period_rejects_axis_unit_mismatch_even_when_predictor_skips(
    export_target: AnalysisTarget,
    calibrated: bool,
    axis_unit: str,
    target_unit: str | None,
) -> None:
    state = aligned_export_state(export_target)
    state.get_spectrum("aligned").unit = axis_unit
    # The planner targets swept, not the spectrum's unnamed source.
    context = ComponentFluxContext(
        FluxFrame(0.2, {}) if calibrated else None,
        "swept",
        {},
    )
    if target_unit is not None:
        context = ComponentFluxContext(
            FluxFrame(0.2, {"swept": DevicePeriod(-8.0, target_unit)})
            if calibrated
            else None,
            "swept",
            {"swept": target_unit},
        )
    service = FitService(state)
    service.configure_flux_context(lambda _model: context)
    service.configure_export(
        lambda _model, points: ComponentExportPlan(
            {"calibration": -points.flux_period},
            FluxWritebackReceipt("written", "swept", None),
        )
    )

    with pytest.raises(
        ValueError, match=f"swept.*{axis_unit}.*{target_unit or 'native'}"
    ):
        service.export_params_receipt()

    model = snapshot(export_target.resource.setup).components["Qcal"].model_dump()
    assert (model["calibration"], model["EJ"]) == (0.0, None)
    assert not export_target.predictor_path.exists()
    assert export_target.resource.ledger.read_events_snapshot() == ()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_invalid_present_other_scalar_fails_before_any_export_output(
    export_target: AnalysisTarget, invalid: float
) -> None:
    state = aligned_export_state(export_target)
    state.get_spectrum("aligned").device_snapshot = {
        "coil": {"value": invalid, "unit": "A"}
    }
    context = ComponentFluxContext(
        FluxFrame(
            0.2, {"unnamed_device": DevicePeriod(8.0), "coil": DevicePeriod(4.0, "A")}
        ),
        "unnamed_device",
        {"unnamed_device": "native", "coil": "A"},
    )
    service = FitService(state)
    service.configure_export(skip_calibration)
    service.configure_flux_context(lambda _model: context)

    with pytest.raises(ValueError, match="coil"):
        service.export_params_receipt()

    assert not export_target.predictor_path.exists()
    assert (
        snapshot(export_target.resource.setup).components["Qcal"].model_dump()["EJ"]
        is None
    )


@pytest.mark.parametrize("recorded", [False, True])
def test_single_device_accepts_absent_or_empty_snapshot(
    export_target: AnalysisTarget, recorded: bool
) -> None:
    state = aligned_export_state(export_target)
    state.get_spectrum("aligned").device_snapshot = {} if recorded else None
    service = FitService(state)
    service.configure_export(skip_calibration)
    service.configure_flux_context(synthetic_flux_context)

    receipt = service.export_params_receipt()

    assert receipt.savepath is not None
    predictor = load_predictor_file(Path(receipt.savepath))
    assert predictor.flux.offset == pytest.approx(0.15)
    assert receipt.predictor_writeback.status == "written"


def test_skipped_predictor_leaves_existing_output_untouched(
    export_target: AnalysisTarget,
) -> None:
    export_target.predictor_path.parent.mkdir(parents=True)
    export_target.predictor_path.write_bytes(b"previous dataset predictor")
    service = FitService(aligned_export_state(export_target))
    service.configure_export(skip_calibration)
    service.configure_flux_context(lambda _model: ComponentFluxContext(None, None, {}))

    receipt = service.export_params_receipt()

    assert receipt.savepath is None
    assert export_target.predictor_path.read_bytes() == b"previous dataset predictor"


@pytest.mark.parametrize("missing", [True, False])
def test_context_provider_failure_precedes_output(
    export_target: AnalysisTarget, missing: bool
) -> None:
    def reject(_model: ComponentSchema) -> ComponentFluxContext:
        raise ValueError("unsupported synthetic kind")

    service = FitService(aligned_export_state(export_target))
    service.configure_export(skip_calibration)
    if not missing:
        service.configure_flux_context(reject)

    with pytest.raises(ValueError, match="Qcal.*qubit/fluxonium/export"):
        service.export_params_receipt()

    assert not export_target.predictor_path.exists()
    assert (
        snapshot(export_target.resource.setup).components["Qcal"].model_dump()["EJ"]
        is None
    )


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
    service.configure_flux_context(synthetic_flux_context)
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
    assert receipt.savepath is not None
    predictor = load_predictor_file(Path(receipt.savepath))
    assert predictor.flux.offset == pytest.approx(0.15)
    assert predictor.flux.period == {"unnamed_device": 2.0}
    assert receipt.params == (5.0, 1.2, 0.4)
    assert target.resource.ledger.get(receipt.analysis_id).event.kind == "analyzed"


def test_export_skip_keeps_calibration_and_writes_energies_and_predictor(
    export_target: AnalysisTarget,
) -> None:
    service = FitService(aligned_export_state(export_target))
    service.configure_export(skip_calibration)
    service.configure_flux_context(synthetic_flux_context)
    receipt = service.export_params_receipt()
    model = snapshot(export_target.resource.setup).components["Qcal"].model_dump()
    assert model["calibration"] == 0.0
    assert (model["EJ"], model["EC"], model["EL"]) == (5.0, 1.2, 0.4)
    assert receipt.flux_writeback.status == "skipped"
    assert receipt.flux_writeback.reason is not None
    assert receipt.savepath is not None
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
