"""Registered opaque run handles cross Run, Analyze and synchronous Save unchanged."""

import json
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import AnalysisDocument, load_run_data
from zcu_tools.experiment.artifacts import AnalysisItem, DataItem
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.entry import AcquiredPayload, ArtifactKey

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._native_save_support import PaneParams, PaneSummary


@dataclass
class OpaqueRun:
    """Adapter-owned handle without the framework's historical cfg/result fields."""

    source: RunRecord[SavingCfg, SavingResult]


class OpaqueAdapter(BaseAdapter[SavingCfg, OpaqueRun, PaneSummary, PaneParams]):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, requires_soc=False, load_data=False
    )
    experiment_tag: ClassVar[str | None] = SavingCore().persistence_spec().tag

    @staticmethod
    def make_form(ctx: FormContext) -> Form[SavingCfg]:
        del ctx
        return Form(SavingCfg, {})

    def run(self, req: RunRequest, cfg: SavingCfg, *, context: RunContext) -> OpaqueRun:
        return OpaqueRun(RunRecord(cfg, SavingCore().run(cfg, context=context)))

    def analyze(
        self, req: AnalyzeRequest[OpaqueRun, PaneParams], *, plots: Plots
    ) -> PaneSummary:
        _, axes = plots.subplots("fit")
        axes.plot(
            req.run_result.source.result.frequency,
            req.run_result.source.result.signal.real,
        )
        return PaneSummary(17.0)

    def data_item(self, key: ArtifactKey, result: object, *, comment: str) -> DataItem:
        if not isinstance(result, OpaqueRun):
            raise TypeError("Expected this adapter's opaque handle")
        return DataItem.from_run(
            key, result.source, spec=SavingCore().persistence_spec(), comment=comment
        )

    def analysis_item(
        self,
        key: ArtifactKey,
        result: object,
        parameters: object | None,
        *,
        pane: Literal["analysis", "post_analysis"],
    ) -> AnalysisItem:
        return AnalysisItem(
            key,
            AnalysisDocument.model_validate(
                {
                    "pane": pane,
                    "parameters": {"captured": 2.0},
                    "summary": {"opaque_mean": 17.0},
                }
            ),
        )


class FailingDataAdapter(OpaqueAdapter):
    def data_item(self, key: ArtifactKey, result: object, *, comment: str) -> DataItem:
        raise ValueError("opaque data preparation rejected")


@pytest.fixture
def opaque_host(
    tmp_path: Path, qapp: QApplication
) -> Generator[tuple[State, Controller]]:
    """Register synthetic adapters in one real offline application host."""
    registry = Registry()
    registry.register("opaque", OpaqueAdapter)
    registry.register("failed_data", FailingDataAdapter)
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    controller = Controller(
        state=state,
        registry=registry,
        view=None,
        bus=BaseEventBus(),
        project_root=str(tmp_path),
    )
    controller.setup_control.new_entry(EntryRequest("opaque-entry"))
    controller.context_control.new_point(
        PointCreateRequest("p", "historical", CommandOrigin("gui"))
    )
    try:
        yield state, controller
    finally:
        for tab_id in tuple(state.list_tab_ids()):
            controller.close_tab(tab_id)
        controller.close()
        qapp.processEvents()


@pytest.mark.parametrize("failed_data", [False, True])
def test_registered_opaque_handle_acquires_analyzes_and_hands_over_items(
    opaque_host: tuple[State, Controller], qapp: QApplication, failed_data: bool
) -> None:
    state, controller = opaque_host
    tab_id = controller.new_tab("failed_data" if failed_data else "opaque")
    controller.run_analyze_control.start_run(tab_id, origin=CommandOrigin("mcp"))
    # The admitted synthetic run returns; pump its terminal owner delivery.
    qt_until(lambda: not state.is_tab_busy(tab_id), label="opaque run terminal")
    tab = state.get_tab(tab_id)
    assert isinstance(tab.run.result, OpaqueRun), (
        "Native evidence must accept an opaque run handle"
    )
    handle = tab.run.result
    metadata = tab.run.metadata
    storage = tab.run.storage
    assert metadata is not None and storage is not None
    acquired = storage.ledger.events()
    assert len(acquired) == 1 and acquired[0].origin == "mcp"
    assert isinstance(acquired[0].payload, AcquiredPayload)
    assert acquired[0].payload.run_id == metadata.run_id
    assert acquired[0].payload.point == "p"
    assert acquired[0].payload.roles == dict(metadata.snapshot.roles)
    assert not (storage.database_path / "runs").exists()

    controller.run_analyze_control.analyze(tab_id, PaneParams())
    # The admitted fit returns and commits its pane on the owner thread.
    qt_until(lambda: not state.is_tab_busy(tab_id), label="opaque analysis terminal")
    assert controller.get_tab_analyze_result(tab_id) == PaneSummary(17.0)
    assert tab.run.result is handle
    controller.context_control.new_point(
        PointCreateRequest("q", "later", CommandOrigin("gui"))
    )
    report = controller.save_control.save_artifacts(
        tab_id, comment="opaque annotation", origin=CommandOrigin("gui")
    )
    data = [item for item in report.outputs if item.key.member == "data"]
    other = [item for item in report.outputs if item.key.member != "data"]
    assert len(data) == 2 and len(other) == 3
    assert all(item.status == ("failed" if failed_data else "saved") for item in data)
    if failed_data:
        assert all(
            item.error and "opaque data preparation rejected" in item.error
            for item in data
        )
    else:
        stored = load_run_data(data[0].output.path)
        assert stored.metadata.run_id == metadata.run_id
        assert stored.metadata.snapshot == metadata.snapshot
        assert stored.payload.metadata.comment == "opaque annotation"
    assert all(item.status == "saved" and item.output.path.is_file() for item in other)
    assert json.loads(
        next(
            item.output.path for item in other if item.output.format == "json"
        ).read_text()
    )["summary"] == {"opaque_mean": 17.0}
    committed = state.get_artifact_snapshots(tab_id)
    projected = {
        (member.key, output.path): (output.status, output.error)
        for section in committed
        for artifact in section.artifacts
        for member in artifact.members
        for output in member.outputs
    }
    assert all(
        projected[(item.key, item.output.path)] == (item.status, item.error)
        for item in report.outputs
    )
    second = controller.save_control.save_artifacts(tab_id, origin=CommandOrigin("mcp"))
    assert second.run_id == report.run_id == metadata.run_id
    assert second.outputs[0].output.path == report.outputs[0].output.path
    if not failed_data:
        assert second.outputs[1].output.path != report.outputs[1].output.path
    assert all(
        item.status == ("failed" if failed_data else "saved")
        for item in second.outputs
        if item.key.member == "data"
    )
    assert all(
        item.status == "saved" for item in second.outputs if item.key.member != "data"
    )
    latest = {
        (member.key, output.path): (output.status, output.error)
        for section in state.get_artifact_snapshots(tab_id)
        for artifact in section.artifacts
        for member in artifact.members
        for output in member.outputs
    }
    assert all(
        latest[(item.key, item.output.path)] == (item.status, item.error)
        for item in second.outputs
    )
    assert [event.origin for event in storage.ledger.events()] == ["mcp", "gui", "mcp"]
    assert tab.run.result is handle
