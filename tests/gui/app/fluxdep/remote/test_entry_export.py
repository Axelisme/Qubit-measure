"""Entry/component selection and predictor writeback through the native RPC seam."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.analysis.flux import ComponentFluxContext, DevicePeriod, FluxFrame
from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
from zcu_tools.gui.app.fluxdep.state import SpectrumEntry
from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.resources.entry import ResultEntry, component_registry, snapshot
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.entry.schema import ComponentSchema
from zcu_tools.resources.predictor_file import load_predictor_file

from tests.gui.app.fluxdep._components import SyntheticFluxonium, synthetic_flux_context
from tests.gui.app.fluxdep._export import aligned_export_state
from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


@pytest.fixture
def analysis_entry(tmp_path: Path) -> Iterator[ResultEntry]:
    component_registry.register("qubit/fluxonium", SyntheticFluxonium)
    try:
        resource = ResultEntry.create(
            "synthetic",
            result_root=tmp_path / "result",
            database_root=tmp_path / "unused",
        )
        resource.setup.add_component("Q1", kind="qubit/fluxonium")
        resource.setup.add_component("Q2", kind="qubit/fluxonium", EJ=9.0)
        resource.save_aliases(EntryAliases(aliases={"qubit": "Q2"}))
        yield resource
    finally:
        component_registry.unregister("qubit/fluxonium")


def test_entry_alias_target_exports_predictor_setup_and_analysis_record(
    route_harness: RouteHarness, analysis_entry: ResultEntry
) -> None:
    route_harness.ctrl.configure_flux_context(synthetic_flux_context)
    client = route_harness.client()
    try:
        opened = route_harness.request(
            client,
            "project.setup",
            project_root=str(analysis_entry.result_path.parent.parent),
            entry="synthetic",
        )
    except RemoteError as error:
        pytest.fail(f"entry/component project inputs were rejected: {error}")
    assert opened["ok"], opened
    info = route_harness.request(client, "project.info")
    assert info["ok"], info
    assert "result" in info
    assert info["result"]["entry"] == "synthetic"
    assert info["result"]["component"] == "Q2"
    assert info["result"]["result_path"] == str(analysis_entry.result_path)

    route_harness.ctrl.state.put_spectrum(
        SpectrumEntry(
            name="calibrated",
            spec_type="TwoTone",
            raw={
                "dev_values": np.array([0.0, 1.0]),
                "fluxs": np.array([0.0, 0.5]),
                "freqs": np.array([4.0, 5.0]),
                "signals": np.ones((2, 2), dtype=complex),
            },
            # Empty sampled sources still supply the first committed calibration.
            points={
                "dev_values": np.array([]),
                "fluxs": np.array([]),
                "freqs": np.array([]),
            },
            flux_half=0.7,
            flux_int=1.7,
            flux_period=2.0,
            aligned=True,
        )
    )
    route_harness.ctrl.state.set_fit_result((5.123456789, 1.2, 0.4))
    assert route_harness.request(client, "fit.result")["ok"]
    assert route_harness.request(client, "spectrum.list")["ok"]
    assert route_harness.request(client, "spectrum.snapshot", name="calibrated")["ok"]
    exported = route_harness.request(client, "fit.export_params")
    assert exported["ok"], exported
    assert "result" in exported
    predictor = load_predictor_file(analysis_entry.predictors_path / "Q2.json")
    assert predictor.model == "fluxonium"
    assert predictor.params.to_dict() == {
        "EJ": 5.123456789,
        "EC": 1.2,
        "EL": 0.4,
    }
    assert predictor.flux.offset == pytest.approx(0.15)
    assert predictor.flux.period == {"unnamed_device": 2.0}
    assert predictor.source.run_id is None
    analysis_entry.setup.refresh()
    components = snapshot(analysis_entry.setup).components
    assert components["Q2"].model_dump() == {
        "kind": "qubit/fluxonium",
        "EJ": 5.123456789,
        "EC": 1.2,
        "EL": 0.4,
    }
    assert components["Q1"].model_dump()["EJ"] is None
    assert exported["result"]["component"] == "Q2"
    assert exported["result"]["flux_writeback"] == {
        "status": "skipped",
        "device": None,
        "reason": "Synthetic target has no calibration. Use fluxdep measurement writeback.",
    }
    assert exported["result"]["predictor_writeback"] == {
        "status": "written",
        "device": "unnamed_device",
        "reason": None,
    }
    record_id = exported["result"]["analysis_id"]
    assert isinstance(record_id, str)
    event = analysis_entry.ledger.get(record_id)
    assert event.event.kind == "analyzed"


def test_skipped_predictor_projects_null_path_and_both_receipts(
    route_harness: RouteHarness, analysis_entry: ResultEntry
) -> None:
    target = AnalysisTarget.open(
        analysis_entry.result_path.parent.parent, "synthetic", "Q2"
    )
    route_harness.ctrl.configure_flux_context(
        lambda _model: ComponentFluxContext(None, None, {})
    )
    client = route_harness.client()
    opened = route_harness.request(
        client,
        "project.setup",
        project_root=str(analysis_entry.result_path.parent.parent),
        entry="synthetic",
        component="Q2",
    )
    assert opened["ok"], opened
    route_harness.ctrl.state.put_spectrum(
        replace(
            aligned_export_state(target).get_spectrum("aligned"),
            filter_mask=None,
            brush_mask=None,
        )
    )
    route_harness.ctrl.state.set_fit_result((5.0, 1.2, 0.4))

    exported = route_harness.request(client, "fit.export_params")

    assert exported["ok"], exported
    assert exported["result"]["savepath"] is None
    assert exported["result"]["predictor_writeback"] == {
        "status": "skipped",
        "device": "unnamed_device",
        "reason": "Component offset is uncalibrated. Calibrate the component before exporting a predictor.",
    }
    assert exported["result"]["flux_writeback"] == {
        "status": "skipped",
        "device": None,
        "reason": "Synthetic target has no calibration. Use fluxdep measurement writeback.",
    }


def test_export_unit_failure_reaches_rpc_without_writing(
    route_harness: RouteHarness, analysis_entry: ResultEntry
) -> None:
    target = AnalysisTarget.open(
        analysis_entry.result_path.parent.parent, "synthetic", "Q2"
    )
    entry = replace(
        aligned_export_state(target).get_spectrum("aligned"),
        filter_mask=None,
        brush_mask=None,
        device_snapshot={"coil": {"value": 1.0, "unit": "mA"}},
    )
    context = ComponentFluxContext(
        FluxFrame(
            0.2, {"unnamed_device": DevicePeriod(8.0), "coil": DevicePeriod(4.0, "A")}
        ),
        "unnamed_device",
        {"unnamed_device": "native", "coil": "A"},
    )
    route_harness.ctrl.configure_flux_context(lambda _model: context)
    client = route_harness.client()
    assert route_harness.request(
        client,
        "project.setup",
        project_root=str(analysis_entry.result_path.parent.parent),
        entry="synthetic",
        component="Q2",
    )["ok"]
    route_harness.ctrl.state.put_spectrum(entry)
    route_harness.ctrl.state.set_fit_result((5.0, 1.2, 0.4))

    exported = route_harness.request(client, "fit.export_params")

    assert not exported["ok"]
    assert exported["error"]["code"] == "controller_error"
    message = exported["error"]["message"]
    assert "coil" in message
    assert "'mA'" in message and "'A'" in message
    assert not target.predictor_path.exists()
    assert snapshot(analysis_entry.setup).components["Q2"].model_dump()["EJ"] == 9.0
    assert analysis_entry.ledger.read_events_snapshot() == ()


@pytest.mark.parametrize(
    ("failure", "diagnostic"),
    [
        ("entry", "missing-entry"),
        ("component", "missing-component"),
        ("kind", "fluxonium"),
        ("alias", "$qubit"),
    ],
)
def test_invalid_entry_target_reports_the_cause_and_preserves_selection(
    route_harness: RouteHarness,
    analysis_entry: ResultEntry,
    failure: str,
    diagnostic: str,
) -> None:
    client = route_harness.client()
    root = str(analysis_entry.result_path.parent.parent)
    try:
        opened = route_harness.request(
            client,
            "project.setup",
            project_root=root,
            entry="synthetic",
            component="Q1",
        )
    except RemoteError as error:
        pytest.fail(f"entry/component project inputs were rejected: {error}")
    assert opened["ok"], opened
    before = route_harness.request(client, "project.info")
    component_registry.register("resonator", ComponentSchema)
    try:
        analysis_entry.setup.add_component("R1", kind="resonator")
        if failure == "alias":
            analysis_entry.save_aliases(EntryAliases(aliases={}))
        entry = "missing-entry" if failure == "entry" else "synthetic"
        component = {"component": "missing-component", "kind": "R1"}.get(failure)
        rejected = route_harness.request(
            client, "project.setup", project_root=root, entry=entry, component=component
        )
        assert "error" in rejected, rejected
        assert diagnostic in rejected["error"]["message"]
        after = route_harness.request(client, "project.info")
        assert "result" in before and "result" in after
        assert after["result"] == before["result"]
    finally:
        component_registry.unregister("resonator")
