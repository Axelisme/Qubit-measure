"""Unit tests for AnalyzeService.

Covers start_analyze, the FIT terminal paths, interactive analyze, and
cancel_interactive — using a real State + real EventBus, with bg mocked.

Stage 2c: AnalyzeService is now an OperationRunner client. _StagedAnalyzeService
no longer has a _submit helper; FIT/post paths use _submit_with_runner internally.
Tests drive terminal paths by capturing bg.last_on_done / on_error.
"""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickAnalysis,
    FluxPickInputs,
    FluxPickState,
)
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepCfg, FluxDepResult
from zcu_tools.experiment.v2.twotone.time_domain.t1 import T1Cfg, T1Result
from zcu_tools.experiment.v2_gui.measure.adapters._support import (
    FluxPickParams,
    FluxPickResult,
)
from zcu_tools.experiment.v2_gui.measure.adapters.onetone.flux_dep import (
    OneToneFluxDepAdapter,
)
from zcu_tools.experiment.v2_gui.measure.adapters.twotone.time_domain.t1 import (
    T1Adapter,
    T1AnalyzeParams,
    T1AnalyzeResult,
)
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest, ContextReadiness
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactKey,
    ArtifactKind,
    SaveStatus,
)
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.interactive import Action, PluginDefinition
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.services.guard import AnalyzePermit
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.event_bus import EventMeta, EventOrigin
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from tests.gui._completion_helpers import on_analyze_failed
from tests.gui._progress_fakes import DirectProgressTransport


def _plots() -> Plots:
    return Plots(NonPresentingHost())


def _make_state(tab_id: str = "tab1") -> State:
    ctx = SessionEnv(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=MagicMock(),
        soccfg=MagicMock(),
        result_dir="/tmp",
        readiness=ContextReadiness.ACTIVE,
    )
    state = State(ctx)
    state.add_tab(
        tab_id,
        Session(adapter_name="fake", adapter=MagicMock(), cfg_schema=MagicMock()),
    )
    # Provide a fake run_result so the tab is not empty
    state.update_tab_result(tab_id, object())
    return state


class _FakeBg:
    """Synchronous background executor stub: captures callbacks for per-test driving."""

    def __init__(self, *, fail_submit: bool = False) -> None:
        self._fail_submit = fail_submit
        self.last_work: Callable[[], Any] | None = None
        self.last_on_done: Callable[[Any], None] | None = None
        self.last_on_error: Callable[[Exception], None] | None = None
        self.submit_count = 0

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if self._fail_submit:
            raise RuntimeError("submit boom")
        self.submit_count += 1
        self.last_work = work
        self.last_on_done = on_done
        self.last_on_error = on_error


def _make_service(
    state: State,
    bus: EventBus,
    *,
    fail_submit: bool = False,
    handles: OperationHandles | None = None,
    writeback: MagicMock | None = None,
) -> tuple[AnalyzeService, _FakeBg]:
    bg = _FakeBg(fail_submit=fail_submit)
    handles = handles or OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    writeback = writeback if writeback is not None else MagicMock()
    writeback.create_draft.return_value = None
    runner = OperationRunner(MagicMock(), handles, progress, bg, bus)  # type: ignore[arg-type]
    svc = AnalyzeService(state, runner, bus, writeback, handles)
    return svc, bg


def _make_two_tab_state() -> State:
    state = _make_state("tab1")
    state.add_tab(
        "tab2",
        Session(adapter_name="fake", adapter=MagicMock(), cfg_schema=MagicMock()),
    )
    state.update_tab_result("tab2", object())
    return state


# ---------------------------------------------------------------------------
# start_analyze — normal path
# ---------------------------------------------------------------------------


def test_start_analyze_submits_to_bg(qapp):
    state = _make_state()
    bus = EventBus()
    svc, bg = _make_service(state, bus)

    svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object(), plots=_plots()
    )

    assert bg.submit_count == 1
    assert state.get_tab("tab1").is_analyzing is True


def test_start_analyze_emits_interaction_event(qapp):
    state = _make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))

    svc, _ = _make_service(state, bus)
    svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object(), plots=_plots()
    )

    assert received == [TabInteractionFact.PRIMARY_ANALYZE_STARTED]


def test_start_analyze_submit_rejection_emits_restore_fact(qapp):
    state = _make_state()
    old_plots = _plots()
    old_plots.adopt("fit", Figure())
    old_plots.finish()
    old_post_plots = _plots()
    old_post_plots.adopt("fit", Figure())
    old_post_plots.finish()
    state.update_tab_analyze("tab1", object(), old_plots)
    state.update_tab_post_analyze("tab1", object(), old_post_plots)
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, _ = _make_service(state, bus, fail_submit=True)

    with pytest.raises(RuntimeError, match="submit boom"):
        svc.start_analyze(
            AnalyzePermit(tab_id="tab1"),
            analyze_params_instance=object(),
            plots=_plots(),
        )

    assert received == [TabInteractionFact.PRIMARY_ANALYZE_START_REJECTED]
    assert state.get_tab("tab1").analysis.plots is old_plots
    assert state.get_tab("tab1").post_analysis.plots is old_post_plots
    assert state.get_tab("tab1").is_analyzing is False


def test_start_analyze_passes_operation_plots_to_worker_adapter(qapp):
    state = _make_state()
    svc, bg = _make_service(state, EventBus())
    plots = _plots()

    svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object(), plots=plots
    )

    assert bg.submit_count == 1
    assert bg.last_work is not None
    result = bg.last_work()
    adapter = state.get_tab("tab1").adapter
    adapter.analyze.assert_called_once()
    assert adapter.analyze.call_args.kwargs == {"plots": plots}
    assert result is adapter.analyze.return_value


# ---------------------------------------------------------------------------
# start_analyze — busy tab rejection
# ---------------------------------------------------------------------------


def test_t1_gui_analysis_publishes_typed_result_and_saveable_named_fit(qapp) -> None:
    state = _make_state()
    tab = state.get_tab("tab1")
    tab.adapter = T1Adapter()
    times = np.linspace(0, 100, 101)
    signals = (0.2 + 0.8 * np.exp(-times / 20)).astype(np.complex128)
    source = RunRecord[T1Cfg, T1Result](cfg=None, result=T1Result(times, signals))
    state.update_tab_result("tab1", source)
    handles = OperationHandles()
    service, bg = _make_service(state, EventBus(), handles=handles)
    plots = _plots()
    token = service.start_analyze(
        AnalyzePermit(tab_id="tab1"), T1AnalyzeParams(skip=3), plots=plots
    )
    assert bg.last_work is not None and bg.last_on_done is not None
    bg.last_on_done(bg.last_work())

    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished"
    pane = state.get_tab("tab1").analysis
    assert isinstance(pane.result, T1AnalyzeResult)
    assert pane.result.t1 == pytest.approx(20, rel=0.01)
    assert pane.plots is plots
    assert tuple(plots) == ("fit",)
    images = [
        item
        for item in state.get_artifact_snapshots("tab1")
        if item.key == ArtifactKey(ArtifactKind.ANALYSIS, "fit")
    ]
    assert len(images) == 1 and images[0].status is SaveStatus.NOT_SAVED
    assert images[0].is_saveable
    output = BytesIO()
    plots["fit"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")
    plots.release()


def _start_onetone_plugin(
    state: State,
    source: RunRecord[FluxDepCfg, FluxDepResult],
    plots: Plots,
    handles: OperationHandles,
    *,
    writeback: MagicMock | None = None,
) -> tuple[AnalyzeService, PluginDefinition[Any, Any], int]:
    adapter = state.get_tab("tab1").adapter
    assert isinstance(adapter, OneToneFluxDepAdapter)
    ctx = state.session_env
    plugin = adapter.make_interactive_plugin(
        AnalyzeRequest(source, FluxPickParams(), ctx.md, ctx.ml, ctx.predictor),
        plots=plots,
    )
    service, _bg = _make_service(
        state, EventBus(), handles=handles, writeback=writeback
    )
    token = service.start_plugin(
        AnalyzePermit(tab_id="tab1"),
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance=FluxPickParams(),
        plots=plots,
    )
    return service, plugin, token


def test_onetone_gui_done_uses_frontend_result_and_retains_named_pick(qapp) -> None:
    state = _make_state()
    tab = state.get_tab("tab1")
    tab.adapter = OneToneFluxDepAdapter()
    values = np.linspace(-0.5, 0.5, 9)
    freqs = np.linspace(4.8, 5.4, 7)
    signals = np.asarray(
        np.sin(values[:, None] * 7 + freqs[None, :] * 9)
        + 1j * np.cos(values[:, None] * 3 - freqs[None, :] * 7),
        dtype=np.complex128,
    )
    source = RunRecord(cfg=None, result=FluxDepResult(values, freqs, signals))
    state.update_tab_result("tab1", source)
    ctx = state.session_env
    ctx.md.flx_half = -0.2
    ctx.md.flx_int = 0.3
    plots = _plots()
    handles = OperationHandles()
    writeback = MagicMock()
    service, plugin, token = _start_onetone_plugin(
        state, source, plots, handles, writeback=writeback
    )
    active = service.get_interactive("tab1")
    assert active is not None
    assert tuple(plots) == ()
    plugin.execute_command(active.session, "swap_lines", {})
    committed = active.session.snapshot()
    assert service.finish_plugin("tab1") is True
    assert service.get_interactive("tab1") is None
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished"
    pane = state.get_tab("tab1").analysis
    assert isinstance(pane.result, FluxPickResult)
    assert pane.result.flx_half == committed.flux_half
    assert pane.result.flx_int == committed.flux_int
    assert pane.result.flx_period == pytest.approx(1.0)
    writeback_items = writeback.create_draft.call_args.args[0]
    assert {item.target_name: item.proposed_value for item in writeback_items} == {
        "flx_half": committed.flux_half,
        "flx_int": committed.flux_int,
        "flx_period": pane.result.flx_period,
    }
    assert pane.plots is plots
    assert tuple(plots) == ("pick",)
    np.testing.assert_allclose(
        np.asarray(plots["pick"].axes[0].lines[0].get_xdata(), dtype=np.float64),
        [committed.flux_half],
    )
    image = [
        item
        for item in state.get_artifact_snapshots("tab1")
        if item.key == ArtifactKey(ArtifactKind.ANALYSIS, "pick")
    ]
    assert len(image) == 1 and image[0].is_saveable
    assert image[0].status is SaveStatus.NOT_SAVED
    plots.release()
    output = BytesIO()
    plots["pick"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")


def test_onetone_gui_cancel_retains_prior_analysis_and_no_new_pick(qapp) -> None:
    state = _make_state()
    tab = state.get_tab("tab1")
    tab.adapter = OneToneFluxDepAdapter()
    previous = object()
    previous_plots = _plots()
    old_figure, _ = previous_plots.subplots("old")
    previous_plots.finish()
    values = np.linspace(-0.5, 0.5, 9)
    freqs = np.linspace(4.8, 5.4, 7)
    source = RunRecord(
        cfg=None,
        result=FluxDepResult(values, freqs, np.ones((9, 7), dtype=np.complex128)),
    )
    state.update_tab_result("tab1", source)
    state.update_tab_analyze("tab1", previous, previous_plots)
    plots = _plots()
    handles = OperationHandles()
    service, _plugin, token = _start_onetone_plugin(state, source, plots, handles)
    assert service.cancel_interactive("tab1") is True
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "cancelled"
    assert service.get_interactive("tab1") is None
    assert state.get_tab("tab1").analysis.result is previous
    assert state.get_tab("tab1").analysis.plots is previous_plots
    assert tuple(plots) == ()
    output = BytesIO()
    old_figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")
    previous_plots.release()


def test_onetone_gui_failed_frontend_analysis_settles_without_replacing_old_pane(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _make_state()
    tab = state.get_tab("tab1")
    tab.adapter = OneToneFluxDepAdapter()
    source = RunRecord(
        cfg=None,
        result=FluxDepResult(
            np.linspace(-0.5, 0.5, 9),
            np.linspace(4.8, 5.4, 7),
            np.ones((9, 7), dtype=np.complex128),
        ),
    )
    state.update_tab_result("tab1", source)
    previous = object()
    previous_plots = _plots()
    old_figure, _ = previous_plots.subplots("old")
    previous_plots.finish()
    state.update_tab_analyze("tab1", previous, previous_plots)
    ctx = state.session_env
    ctx.md.flx_half = -0.2
    ctx.md.flx_int = 0.3
    plots = _plots()

    def fail_kernel(
        inputs: FluxPickInputs, committed: FluxPickState
    ) -> FluxPickAnalysis:
        np.testing.assert_array_equal(inputs.signals, source.result.signals)
        assert committed.flux_half == pytest.approx(-0.2)
        raise RuntimeError("frontend analysis failed")

    monkeypatch.setattr(
        "zcu_tools.experiment.v2_gui.measure.adapters._support.flux_pick_plugin.analyze_flux_pick",
        fail_kernel,
    )
    handles = OperationHandles()
    service, _plugin, token = _start_onetone_plugin(state, source, plots, handles)
    assert service.finish_plugin("tab1") is True
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "failed"
    assert service.get_interactive("tab1") is None
    assert state.get_tab("tab1").analysis.result is previous
    assert state.get_tab("tab1").analysis.plots is previous_plots
    assert tuple(plots) == ()
    image = BytesIO()
    old_figure.savefig(image, format="png")
    assert image.getvalue().startswith(b"\x89PNG")
    previous_plots.release()


def test_start_analyze_rejects_busy_tab(qapp):
    state = _make_state()
    state.set_tab_running("tab1", True)
    svc, _ = _make_service(state, EventBus())

    with pytest.raises(FailedPreconditionError, match="busy") as exc_info:
        svc.start_analyze(
            AnalyzePermit(tab_id="tab1"),
            analyze_params_instance=object(),
            plots=_plots(),
        )

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""


# ---------------------------------------------------------------------------
# on_terminal (FIT) — finish / fail paths via bg callbacks
# ---------------------------------------------------------------------------


def test_on_analyze_finished_updates_state(qapp):
    state = _make_state()
    bus = EventBus()
    svc, bg = _make_service(state, bus)

    token = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object(), plots=_plots()
    )

    fake_result = object()

    finished_signals: list = []
    bus.subscribe(
        TabInteractionChangedPayload,
        lambda payload: (
            finished_signals.append(
                (payload.tab_id, state.get_tab(payload.tab_id).analysis.result)
            )
            if payload.fact is TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED
            else None
        ),
    )

    # Trigger the on_done path (runner calls on_terminal with ok=True)
    assert bg.last_on_done is not None
    bg.last_on_done(fake_result)

    assert state.get_tab("tab1").analysis.result is fake_result
    assert state.get_tab("tab1").is_analyzing is False
    assert len(finished_signals) == 1
    assert finished_signals[0] == ("tab1", fake_result)
    # Handle settles on terminal
    outcome = svc._handles.poll(token)
    assert outcome is not None and outcome.status == "finished"


def test_on_analyze_finished_emits_interaction_event(qapp):
    state = _make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, bg = _make_service(state, bus)

    svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object(), plots=_plots()
    )

    fake_result = object()
    assert bg.last_on_done is not None
    bg.last_on_done(fake_result)

    assert received == [
        TabInteractionFact.PRIMARY_ANALYZE_STARTED,
        TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED,
    ]


# ---------------------------------------------------------------------------
# Interactive analysis (no worker; result produced on the user's Done)
# ---------------------------------------------------------------------------


def test_framework_session_drives_committed_result_and_one_terminal_operation(qapp):
    state = _make_state()
    handles = OperationHandles()
    svc, bg = _make_service(state, EventBus(), handles=handles)

    def require_pick(position: int) -> None:
        if position == 0:
            raise FailedPreconditionError("pick a line")

    plugin = PluginDefinition(
        "pick",
        0,
        (),
        require_pick,
        lambda position: MagicMock(position=position),
    )

    token = svc.start_plugin(
        AnalyzePermit(tab_id="tab1"),
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=_plots(),
    )
    active = svc.get_interactive("tab1")
    assert active is not None and active.plugin is plugin
    Action[int, int](lambda current, position: position).execute(active.session, 3)
    assert active.session.snapshot() == 3
    assert bg.submit_count == 0

    svc.finish_plugin("tab1")
    result = state.get_tab("tab1").analysis.result
    assert result is not None
    assert result.position == 3
    assert state.get_tab("tab1").is_analyzing is False
    assert svc.get_interactive("tab1") is None
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished"
    svc.finish_plugin("tab1")
    assert state.get_tab("tab1").analysis.result is result


def test_unavailable_finish_keeps_session_active_and_cancel_discards_late_done(qapp):
    state = _make_state()
    handles = OperationHandles()
    svc, _ = _make_service(state, EventBus(), handles=handles)

    def require_pick(position: int) -> None:
        if position == 0:
            raise FailedPreconditionError("pick a line")

    plugin = PluginDefinition("pick", 0, (), require_pick, lambda position: MagicMock())
    token = svc.start_plugin(
        AnalyzePermit(tab_id="tab1"),
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=_plots(),
    )
    active = svc.get_interactive("tab1")
    assert active is not None
    with pytest.raises(FailedPreconditionError, match="pick a line"):
        svc.finish_plugin("tab1")
    assert svc.get_interactive("tab1") is active
    assert state.get_tab("tab1").is_analyzing is True

    assert svc.cancel_interactive("tab1") is True
    svc.finish_plugin("tab1")
    assert state.get_tab("tab1").analysis.result is None
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "cancelled"
    with pytest.raises(FailedPreconditionError):
        active.session.snapshot()


def test_result_build_failure_settles_failed_without_reopening_input(qapp):
    state = _make_state()
    handles = OperationHandles()
    svc, _ = _make_service(state, EventBus(), handles=handles)

    def fail_result(position: int) -> object:
        raise RuntimeError("cannot build result")

    plugin = PluginDefinition("pick", 1, (), lambda position: None, fail_result)
    token = svc.start_plugin(
        AnalyzePermit(tab_id="tab1"),
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=_plots(),
    )
    active = svc.get_interactive("tab1")
    assert active is not None
    svc.finish_plugin("tab1")
    assert state.get_tab("tab1").is_analyzing is False
    assert state.get_tab("tab1").analysis.result is None
    assert svc.get_interactive("tab1") is None
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "failed"
    with pytest.raises(FailedPreconditionError):
        active.session.commit(lambda current: 2)


def test_plugin_record_failure_settles_failed_and_keeps_previous_result(qapp):
    state = _make_state()
    previous = MagicMock()
    previous_plots = _plots()
    previous_plots.adopt("fit", Figure())
    previous_plots.finish()
    state.update_tab_analyze("tab1", previous, previous_plots)
    handles = OperationHandles()
    bus = EventBus()
    svc, _ = _make_service(state, bus, handles=handles)
    plugin = PluginDefinition(
        "pick",
        2,
        (),
        lambda _state: None,
        lambda value: MagicMock(value=value),
    )
    token = svc.start_plugin(
        AnalyzePermit(tab_id="tab1"),
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=_plots(),
    )
    active = svc.get_interactive("tab1")
    assert active is not None
    writeback = svc._writeback
    assert isinstance(writeback, MagicMock)
    writeback.create_draft.side_effect = RuntimeError("writeback boom")

    assert svc.finish_plugin("tab1") is True

    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "failed"
    assert state.get_tab("tab1").analysis.result is previous
    assert state.get_tab("tab1").analysis.plots is previous_plots
    assert not state.get_tab("tab1").is_analyzing
    assert svc.get_interactive("tab1") is None
    with pytest.raises(FailedPreconditionError):
        active.session.snapshot()


def test_concurrent_plugin_tabs_keep_independent_sessions_and_handles(qapp):
    state = _make_two_tab_state()
    handles = OperationHandles()
    svc, _ = _make_service(state, EventBus(), handles=handles)
    plugin = PluginDefinition(
        "pick",
        0,
        (),
        lambda _state: None,
        lambda position: MagicMock(position=position),
    )
    owner = ManualOwnerScheduler()
    first = svc.start_plugin(
        AnalyzePermit(tab_id="tab1"),
        plugin,
        owner,
        analyze_params_instance="submitted",
        plots=_plots(),
    )
    second = svc.start_plugin(
        AnalyzePermit(tab_id="tab2"),
        plugin,
        owner,
        analyze_params_instance="submitted",
        plots=_plots(),
    )
    tab1 = svc.get_interactive("tab1")
    tab2 = svc.get_interactive("tab2")
    assert tab1 is not None and tab2 is not None
    Action[int, int](lambda _current, position: position).execute(tab1.session, 5)
    assert tab2.session.snapshot() == 0
    assert svc.cancel_interactive("tab1") is True
    first_outcome = handles.poll(first)
    assert first_outcome is not None and first_outcome.status == "cancelled"
    assert handles.poll(second) is None
    assert svc.get_interactive("tab2") is tab2
    Action[int, int](lambda _current, position: position).execute(tab2.session, 8)
    assert svc.finish_plugin("tab2") is True
    result = state.get_tab("tab2").analysis.result
    assert result is not None and result.position == 8
    assert state.get_tab("tab1").analysis.result is None
    second_outcome = handles.poll(second)
    assert second_outcome is not None and second_outcome.status == "finished"


def test_start_plugin_rejects_busy_tab(qapp):
    state = _make_state()
    state.set_tab_running("tab1", True)
    svc, _ = _make_service(state, EventBus())
    plugin = PluginDefinition("pick", 0, (), lambda _state: None, lambda state: state)

    with pytest.raises(FailedPreconditionError, match="busy"):
        svc.start_plugin(
            AnalyzePermit(tab_id="tab1"),
            plugin,
            ManualOwnerScheduler(),
            analyze_params_instance="submitted",
            plots=_plots(),
        )
    assert svc.get_interactive("tab1") is None


def test_interactive_start_and_finish_keep_captured_operation_origin(
    qapp,
) -> None:
    state = _make_state()
    bus = EventBus()
    svc, _ = _make_service(state, bus)
    observed: list[tuple[TabInteractionFact, EventMeta]] = []
    bus.subscribe_with_meta(
        TabInteractionChangedPayload,
        lambda payload, meta: observed.append((payload.fact, meta)),
    )
    plugin = PluginDefinition(
        "pick", 2, (), lambda _state: None, lambda state: MagicMock()
    )
    with bus.origin(EventOrigin(kind="agent", client_id="client-a")):
        token = svc.start_plugin(
            AnalyzePermit(tab_id="tab1"),
            plugin,
            ManualOwnerScheduler(),
            analyze_params_instance="submitted",
            plots=_plots(),
        )
    svc.finish_plugin("tab1")

    assert [fact for fact, _meta in observed] == [
        TabInteractionFact.PRIMARY_ANALYZE_STARTED,
        TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED,
    ]
    assert [meta.origin for _fact, meta in observed] == [
        EventOrigin(kind="agent", client_id="client-a", operation_id=str(token)),
        EventOrigin(kind="agent", client_id="client-a", operation_id=str(token)),
    ]


# ---------------------------------------------------------------------------
# cancel_interactive — agent-side settle of an in-flight interactive picker
# ---------------------------------------------------------------------------


def test_cancel_interactive_clears_analyzing_and_settles_cancelled(qapp):
    state = _make_state()
    bus = EventBus()
    svc, _ = _make_service(state, bus)
    handles = OperationHandles()
    svc, _ = _make_service(state, bus, handles=handles)
    plugin = PluginDefinition(
        "pick", 0, (), lambda _state: None, lambda state: MagicMock()
    )
    token = svc.start_plugin(
        AnalyzePermit(tab_id="tab1"),
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=_plots(),
    )
    assert state.get_tab("tab1").is_analyzing is True

    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    # A cancel is not a failure: analyze_failed must NOT fire (no error dialog).
    failed: list = []
    on_analyze_failed(svc, lambda tid, err: failed.append((tid, err)))

    cancelled = svc.cancel_interactive("tab1")

    assert cancelled is True
    # is_analyzing cleared -> the tab is no longer busy, so it can be closed.
    assert state.get_tab("tab1").is_analyzing is False
    assert state.is_tab_busy("tab1") is False
    # The handle settles cancelled (the agent's analyze-poll resolves), exactly once.
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "cancelled"
    assert handles.live_count() == 0
    assert "tab1" not in svc._active_tokens
    # Interaction event fired; no failure signal.
    assert received == [TabInteractionFact.PRIMARY_ANALYZE_CANCELLED]
    assert failed == []


def test_cancel_interactive_no_inflight_is_graceful_noop(qapp):
    state = _make_state()
    svc, _ = _make_service(state, EventBus())

    # No analyze started at all -> graceful no-op, no raise, no state change.
    assert svc.cancel_interactive("tab1") is False
    assert state.get_tab("tab1").is_analyzing is False
    assert svc._handles.live_count() == 0


def test_cancel_interactive_does_not_touch_fit_analyze(qapp):
    state = _make_state()
    svc, bg = _make_service(state, EventBus())
    handles = svc._handles

    # A worker-backed FIT analyze is in flight (its callback will settle it later).
    token = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object(), plots=_plots()
    )
    assert bg.submit_count == 1

    # cancel_interactive must not reach into a FIT analyze (settling its handle
    # while the worker callback is still pending would double-settle on finish).
    assert svc.cancel_interactive("tab1") is False
    assert state.get_tab("tab1").is_analyzing is True  # FIT analyze untouched
    assert handles.poll(token) is None  # still live, settles via its own callback


def test_cancel_interactive_is_idempotent(qapp):
    state = _make_state()
    svc, _ = _make_service(state, EventBus())
    plugin = PluginDefinition(
        "pick", 0, (), lambda _state: None, lambda state: MagicMock()
    )
    svc.start_plugin(
        AnalyzePermit(tab_id="tab1"),
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=_plots(),
    )

    assert svc.cancel_interactive("tab1") is True
    # A second cancel finds nothing in flight -> graceful no-op.
    assert svc.cancel_interactive("tab1") is False


def test_finish_after_cancel_interactive_is_inert(qapp):
    # Defensive: if a late Done arrives after a cancel, the already-settled token
    # is gone, so finish_interactive must not resurrect the tab into analyzing.
    state = _make_state()
    svc, _ = _make_service(state, EventBus())
    plugin = PluginDefinition(
        "pick", 0, (), lambda _state: None, lambda state: MagicMock()
    )
    svc.start_plugin(
        AnalyzePermit(tab_id="tab1"),
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=_plots(),
    )
    svc.cancel_interactive("tab1")
    assert svc.finish_plugin("tab1") is False
    assert state.get_tab("tab1").analysis.result is None
    assert state.get_tab("tab1").is_analyzing is False


# ---------------------------------------------------------------------------
# Background analyze failure
# ---------------------------------------------------------------------------


def test_background_analyze_failure_resets_state(qapp):
    state = _make_state()
    bus = EventBus()
    svc, bg = _make_service(state, bus)

    token = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object(), plots=_plots()
    )
    assert state.get_tab("tab1").is_analyzing is True

    failed_signals: list = []
    on_analyze_failed(svc, lambda tid, err: failed_signals.append((tid, err)))

    error = RuntimeError("analysis failed")
    assert bg.last_on_error is not None
    bg.last_on_error(error)

    assert state.get_tab("tab1").is_analyzing is False
    assert len(failed_signals) == 1
    outcome = svc._handles.poll(token)
    assert outcome is not None and outcome.status == "failed"


def test_background_analyze_failure_emits_interaction_event(qapp):
    state = _make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, bg = _make_service(state, bus)

    svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object(), plots=_plots()
    )
    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("oops"))

    assert received == [
        TabInteractionFact.PRIMARY_ANALYZE_STARTED,
        TabInteractionFact.PRIMARY_ANALYZE_FAILED,
    ]


# ---------------------------------------------------------------------------
# Concurrent tabs — no exclusion gate (ADR-0066): each settles its own token
# ---------------------------------------------------------------------------


def test_two_tabs_settle_their_own_tokens(qapp):
    state = _make_two_tab_state()
    svc, bg = _make_service(state, EventBus())
    handles = svc._handles

    token1 = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object(), plots=_plots()
    )
    # Save tab1's on_done before tab2 overwrites it in the bg stub
    on_done_1 = bg.last_on_done

    token2 = svc.start_analyze(
        AnalyzePermit(tab_id="tab2"), analyze_params_instance=object(), plots=_plots()
    )
    on_done_2 = bg.last_on_done

    # Two distinct live handles — the second start must not clobber the first.
    assert token1 != token2
    assert handles.poll(token1) is None  # still pending
    assert handles.poll(token2) is None
    assert handles.live_count() == 2

    # Finish tab1 only: its token settles, tab2's stays live.
    r1 = object()
    assert on_done_1 is not None
    on_done_1(r1)

    outcome1 = handles.poll(token1)
    assert outcome1 is not None and outcome1.status == "finished"
    assert handles.poll(token2) is None  # tab2 untouched
    assert state.get_tab("tab1").is_analyzing is False
    assert state.get_tab("tab2").is_analyzing is True
    assert handles.live_count() == 1

    # Finish tab2: its own (later) token settles.
    r2 = object()
    assert on_done_2 is not None
    on_done_2(r2)

    outcome2 = handles.poll(token2)
    assert outcome2 is not None and outcome2.status == "finished"
    assert handles.live_count() == 0
    assert "tab2" not in svc._active_tokens


# ---------------------------------------------------------------------------
# Terminal slot post-processing raises — tab cleared, handle settled failed
# ---------------------------------------------------------------------------


def test_on_analyze_finished_post_processing_raise_settles_failed(qapp):
    state = _make_state()
    svc, bg = _make_service(state, EventBus())
    handles = svc._handles

    token = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object(), plots=_plots()
    )

    # Make the service-side post-processing raise (writeback compute blows up).
    boom = RuntimeError("writeback boom")
    writeback = svc._writeback
    assert isinstance(writeback, MagicMock)
    writeback.create_draft.side_effect = boom

    failed: list = []
    on_analyze_failed(svc, lambda tid, err: failed.append((tid, err)))

    result = object()
    # Must not raise out of the slot (would crash Qt).
    assert bg.last_on_done is not None
    bg.last_on_done(result)

    # Tab cleared, handle settled failed, failure signal emitted — mirroring the
    # worker-side _failed path.
    assert state.get_tab("tab1").is_analyzing is False
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == "failed"
    assert outcome.error == str(boom)
    assert failed == [("tab1", str(boom))]
    assert handles.live_count() == 0
