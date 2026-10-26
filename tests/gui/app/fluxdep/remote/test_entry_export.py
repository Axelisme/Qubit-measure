"""Entry/component selection and predictor writeback through the native RPC seam."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.gui.app.fluxdep.state import SpectrumEntry
from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.resources.entry import ResultEntry, component_registry, snapshot
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.entry.schema import ComponentSchema
from zcu_tools.resources.predictor_file import load_predictor_file

from tests.gui.app.fluxdep._components import SyntheticFluxonium
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
        "flux_half": 0.7,
        "flux_period": 2.0,
        "flux_bias": 0.0,
    }
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
    record_id = exported["result"]["analysis_id"]
    assert isinstance(record_id, str)
    event = analysis_entry.ledger.get(record_id)
    assert event.event.kind == "analyzed"


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
