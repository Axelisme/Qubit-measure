"""Production Controller Run, portable Save/Load and recorded analysis context."""

from collections.abc import Generator
from dataclasses import dataclass, replace
from pathlib import Path
from shutil import copyfile, rmtree
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import PulseSpec
from zcu_tools.datafile import load, save
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    PostAnalysis,
    PostAnalyzeRequest,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.resources.entry import AcquiredPayload, ComponentSchema, ResultEntry

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._native_save_support import PaneParams, SavingAdapter
from tests.gui.app.measure._record_support import (
    require_metadata,
    require_record,
    require_target,
)
from tests.resources.entry.view_support import registered_model

pytestmark = pytest.mark.uses_wall_clock


class EnvironmentSchema(ComponentSchema):
    rate: float = 1.0


@dataclass(frozen=True)
class EnvironmentSummary:
    rate: float
    gain: float | None


def inspect_environment(parameters: ParameterContext | None) -> EnvironmentSummary:
    assert parameters is not None
    if not parameters.document().components:
        return EnvironmentSummary(-1.0, None)
    rate = parameters.value("$driver.rate")
    assert isinstance(rate, float)
    pulse = parameters.library.module_spec("Q1.pulses.pi")
    assert isinstance(pulse, PulseSpec)
    gain = pulse.gain
    assert isinstance(gain, float)
    return EnvironmentSummary(rate, gain)


def analyze_environment(
    request: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], PaneParams],
    *,
    plots: Plots,
) -> EnvironmentSummary:
    plots.subplots(num="recorded")
    return inspect_environment(request.parameters)


def post_environment(
    request: PostAnalyzeRequest[
        RunRecord[SavingCfg, SavingResult], EnvironmentSummary, PaneParams
    ],
    *,
    plots: Plots,
) -> EnvironmentSummary:
    plots.subplots(num="recorded-post")
    result = inspect_environment(request.parameters)
    assert request.analyze_result == result
    return result


@pytest.fixture
def snapshot_controller(
    tmp_path: Path, qapp: QApplication, request: pytest.FixtureRequest
) -> Generator[tuple[Controller, State]]:
    with registered_model("snapshot-flow/environment", EnvironmentSchema):
        registry = Registry()
        registry.register(
            "snapshot",
            Adapter(
                spec=SavingAdapter.spec,
                make_form=SavingAdapter.make_form,
                run=SavingAdapter.run,
                guide=None,
                analysis=Analysis(
                    fn=analyze_environment, options=lambda source, env: PaneParams()
                ),
                post_analysis=PostAnalysis(
                    fn=post_environment,
                    options=lambda source, result, env: PaneParams(),
                ),
            ),
        )
        state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
        controller = Controller(
            state, registry, None, BaseEventBus(), project_root=str(tmp_path)
        )
        controller.setup_control.new_entry(EntryRequest("recorded"))
        global_context = getattr(request, "param", "point") == "global"
        if not global_context:
            controller.setup_control.new_point(
                PointCreateRequest("p", "captured", CommandOrigin("gui"))
            )
        context = controller.context_control
        context.add_component(
            "Q1", "snapshot-flow/environment", {"rate": 7.0}, setup=global_context
        )
        if global_context:
            controller.setup_control.open_entry(EntryRequest("recorded"))
        context.set_module(
            require_target(context.read_context()),
            "Q1.pulses.pi",
            PulseSpec(
                ch=0,
                nqz=1,
                freq=100.0,
                gain=0.25,
                waveform="const",
            ),
        )
        context.apply_alias_rows((AliasRow("driver", "Q1"),))
        controller.session_services.soc_connection.install_soc(
            MagicMock(), make_mock_soccfg()
        )
        try:
            yield controller, state
        finally:
            controller.close()
            qapp.processEvents()


def finish(controller: Controller, state: State, tab: str, token: int) -> None:
    qt_until(lambda: not state.is_tab_busy(tab), label="snapshot flow terminal")
    outcome = controller.operation_control.await_operation(token, 0.0)
    assert outcome.outcome is not None and outcome.outcome.status == "finished", outcome


@pytest.mark.parametrize("snapshot_controller", ["global"], indirect=True)
def test_global_controller_run_analyze_and_save_preserve_entry_capture(
    snapshot_controller: tuple[Controller, State], tmp_path: Path
) -> None:
    controller, state = snapshot_controller
    assert controller.get_session_env().readiness.value == "global"
    tab_id = controller.tab_control.new_tab("snapshot")
    finish(controller, state, tab_id, controller.run_analyze_control.start_run(tab_id))
    tab = state.get_tab(tab_id)
    metadata = require_metadata(tab.run)
    assert metadata.snapshot.entry_name == "recorded"
    assert metadata.snapshot.point is None
    assert tab.run.storage is not None and tab.run.storage.point is None
    entry = ResultEntry.open(
        "recorded", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    events = entry.ledger.events()
    assert len(events) == 1 and events[0].kind == "acquired"
    assert isinstance(events[0].payload, AcquiredPayload)
    assert events[0].payload.point is None
    assert events[0].payload.run_id == metadata.run_id
    controller.context_control.set_parameter(
        require_target(controller.context_control.read_context()), "Q1.rate", 99.0
    )
    finish(
        controller,
        state,
        tab_id,
        controller.run_analyze_control.analyze(tab_id, PaneParams()),
    )
    assert tab.analysis.result == EnvironmentSummary(7.0, 0.25)
    preview = {
        output.path
        for section in state.get_artifact_snapshots(tab_id)
        for artifact in section.artifacts
        for member in artifact.members
        for output in member.outputs
    }
    assert preview and not any(path.exists() for path in preview)
    report = controller.save_control.save_artifacts(tab_id, origin=CommandOrigin("gui"))
    assert all(item.status == "saved" for item in report.outputs)
    assert {item.output.path for item in report.outputs} == preview
    copies = [
        item.output.path
        for item in report.outputs
        if item.output.format == "png"
        and item.output.path.is_relative_to(entry.result_path)
    ]
    assert copies and all(
        path.parent == entry.result_path / "figures" for path in copies
    )
    assert [event.kind for event in entry.ledger.events()] == ["acquired", "saved"]
    index = controller.query_results()
    assert len(index.runs) == 1
    assert index.runs[0].point is None and index.runs[0].registered
    assert index.runs[0].run_id == metadata.run_id


@pytest.mark.parametrize("legacy", [False, True])
def test_controller_run_save_load_analyze_and_post_need_no_original_point(
    snapshot_controller: tuple[Controller, State],
    tmp_path: Path,
    legacy: bool,
) -> None:
    controller, state = snapshot_controller
    tab_id = controller.tab_control.new_tab("snapshot")
    finish(controller, state, tab_id, controller.run_analyze_control.start_run(tab_id))
    tab = state.get_tab(tab_id)
    metadata = require_metadata(tab.run)
    assert metadata.parameter_snapshot is not None
    assert metadata.snapshot.entry_name == "recorded" and metadata.snapshot.point == "p"
    controller.context_control.set_parameter(
        require_target(controller.context_control.read_context()), "Q1.rate", 99.0
    )
    report = controller.save_control.save_data(tab_id, origin=CommandOrigin("gui"))
    native = next(item for item in report.outputs if item.output.format == "data_h5")
    assert native.status == "saved", native.error
    portable = tmp_path / "arbitrary" / "portable.h5"
    portable.parent.mkdir()
    copyfile(native.output.path, portable)
    if legacy:
        stored = load(portable)
        document = dict(stored.docs["metadata"])
        document.pop("parameter_snapshot")
        save(
            portable,
            replace(stored, docs={**stored.docs, "metadata": document}),
            replace=True,
        )
    controller.session_services.context.clear_context()
    rmtree(tmp_path / "result" / "recorded")
    outcome = controller.run_analyze_control.load_tab_result(tab_id, str(portable))
    assert outcome.analysis_error is None
    assert tab.run.storage is None and tab.run.parameters is not None
    assert tab.run.parameters.target is None
    expected = (
        EnvironmentSummary(-1.0, None) if legacy else EnvironmentSummary(7.0, 0.25)
    )
    finish(
        controller,
        state,
        tab_id,
        controller.run_analyze_control.analyze(tab_id, PaneParams()),
    )
    assert tab.analysis.result == expected
    finish(
        controller,
        state,
        tab_id,
        controller.run_analyze_control.start_post_analyze(tab_id, PaneParams()),
    )
    assert tab.post_analysis.result == expected
    complete = controller.save_control.save_artifacts(
        tab_id, origin=CommandOrigin("gui")
    )
    assert all(item.status == "saved" for item in complete.outputs)
    rewritten = next(
        item.output.path for item in complete.outputs if item.output.format == "data_h5"
    )
    assert rewritten == native.output.path
    np.testing.assert_array_equal(
        next(iter(load(rewritten).vars.values())).values,
        require_record(tab.run).result.signal,
    )
    assert list(rewritten.parent.glob("analysis/*.json"))
    assert list(rewritten.parent.glob("figures/*.png"))
    assert list((tmp_path / "result/recorded/points/p/figures").glob("*.png"))
