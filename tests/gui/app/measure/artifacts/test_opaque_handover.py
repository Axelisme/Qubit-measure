"""RunRecord envelopes carry adapter-owned opaque Results through host handovers."""

import json
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import load
from zcu_tools.experiment.artifacts import AnalysisDocument, AnalysisItem, DataItem
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    RunRequest,
    SessionEnv,
)
from zcu_tools.experiment import AxesSpec, Axis, ZSpec
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis
import numpy as np
from numpy.typing import NDArray
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
from tests.gui.app.measure._record_support import require_record


@dataclass
class OpaqueRun:
    """Adapter-owned handle without the framework's historical cfg/result fields."""

    frequency: NDArray[np.float64]
    signal: NDArray[np.complex128]

    def __init__(self, source: SavingResult) -> None:
        self.frequency = source.frequency
        self.signal = source.signal


def opaque_form(ctx: FormContext) -> Form[SavingCfg]:
    return Form(SavingCfg, {})


def opaque_run(cfg: SavingCfg, *, context: RunContext) -> OpaqueRun:
    return OpaqueRun(SavingCore().run(cfg, context=context))


def opaque_analyze(
    req: AnalyzeRequest[RunRecord[SavingCfg, OpaqueRun], PaneParams], *, plots: Plots
) -> PaneSummary:
    _, axes = plots.subplots("fit")
    axes.plot(req.run_result.result.frequency, req.run_result.result.signal.real)
    return PaneSummary(17.0)


OpaqueAdapter = Adapter(
    spec=AxesSpec(
        axes=(Axis("frequency", "frequency", "Hz", scale=1e6),),
        z=ZSpec("signal", "signal", "V"),
        result_type=OpaqueRun,
        cfg_type=SavingCfg,
        tag="test/opaque",
    ),
    make_form=opaque_form,
    run=opaque_run,
    guide=None,
    analysis=Analysis(fn=opaque_analyze, options=lambda source, env: PaneParams()),
)


@pytest.fixture
def opaque_host(
    tmp_path: Path, qapp: QApplication
) -> Generator[tuple[State, Controller]]:
    """Register synthetic adapters in one real offline application host."""
    registry = Registry()
    registry.register("opaque", OpaqueAdapter)
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


def test_registered_opaque_handle_acquires_analyzes_and_hands_over_items(
    opaque_host: tuple[State, Controller], qapp: QApplication
) -> None:
    state, controller = opaque_host
    tab_id = controller.new_tab("opaque")
    controller.run_analyze_control.start_run(tab_id, origin=CommandOrigin("mcp"))
    # The admitted synthetic run returns; pump its terminal owner delivery.
    qt_until(lambda: not state.is_tab_busy(tab_id), label="opaque run terminal")
    tab = state.get_tab(tab_id)
    assert isinstance(require_record(tab.run).result, OpaqueRun), (
        "The record envelope must preserve the adapter's opaque Result"
    )
    handle = tab.run.result
    metadata = require_record(tab.run).metadata
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
    assert all(item.status == "saved" for item in data)
    stored = load(data[0].output.path)
    assert metadata_from_container(stored).run_id == metadata.run_id
    assert metadata_from_container(stored).snapshot == metadata.snapshot
    assert all(item.status == "saved" and item.output.path.is_file() for item in other)
    assert json.loads(
        next(
            item.output.path for item in other if item.output.format == "json"
        ).read_text()
    )["summary"] == {"mean": 17.0}
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
    assert second.outputs[1].output.path != report.outputs[1].output.path
    assert all(
        item.status == "saved" for item in second.outputs if item.key.member == "data"
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
