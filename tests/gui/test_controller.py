"""Controller skeleton (tab + run flow)."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from tempfile import mkdtemp
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.native_persistence import RestoredRun
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    ContextReadiness,
    NoAnalyzeParams,
    SessionEnv,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.measure.events.tab import (
    TabContentChangedPayload,
    TabContentFact,
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services import (
    ConnectionPreferences,
)
from zcu_tools.gui.app.measure.services import (
    create_persistence_caretaker as PersistenceCaretaker,
)
from zcu_tools.gui.app.measure.services.ports import RestoreIssue, RestoreReport
from zcu_tools.gui.app.measure.state import DeviceStatus, State
from zcu_tools.gui.event_bus import BaseEventBus, EventMeta, EventOrigin
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.device_models import (
    ConnectDeviceRequest,
    DisconnectDeviceRequest,
)
from zcu_tools.gui.session.ports import (
    BackgroundAdmissionClosedError,
    OperationConflictError,
    OperationKind,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2.mocksoc import make_mock_soccfg

from tests.gui._controller_adapter import (
    InteractiveAdapter,
    RecordingAdapter,
    RecordingCfg,
)
from tests.gui._qt_waits import qt_until as _wait_for
from tests.gui.app.measure._load_support import write_native_file

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_ctx() -> SessionEnv:
    return SessionEnv(
        parameters=None,
        soc=MagicMock(),  # simulate connected soc
        soccfg=make_mock_soccfg(),
        res_name="fake_res",
    )


def _make_view() -> MagicMock:
    view = MagicMock()
    view.show_status_message = MagicMock()
    view.show_error_dialog = MagicMock()
    view.make_run_container = MagicMock(return_value=None)

    # The Controller fans diagnostics out via notify_diagnostic (ADR-0068);
    # mirror MainWindow's dispatch so tests can assert on show_*.
    def _notify(severity: str, title: str, message: str) -> None:
        if severity == "error":
            view.show_error_dialog(title or "Error", message)
        else:
            view.show_status_message(message)

    view.notify_diagnostic = MagicMock(side_effect=_notify)
    return view


class ControllerFixture:
    """Holds all objects to prevent premature GC during tests."""

    def __init__(
        self, cache_dir=None, project_root=None, *, mock_emit: bool = True
    ) -> None:
        self.state = State(_make_ctx())
        self.registry = Registry()
        self.registry.register("fake", RecordingAdapter)
        self.registry.register("interactive", InteractiveAdapter)
        self.view = _make_view()
        from zcu_tools.gui.event_bus import BaseEventBus

        self.bus = BaseEventBus()
        if mock_emit:
            self.bus.emit = MagicMock(wraps=self.bus.emit)
        self.ctrl = Controller(
            state=self.state,
            registry=self.registry,
            view=self.view,
            bus=self.bus,
            project_root=project_root or mkdtemp(prefix="context-"),
        )
        self.ctrl.setup_control.new_entry(EntryRequest("fake_qubit"))
        self.ctrl.setup_control.new_point(
            PointCreateRequest("ctx001", "", CommandOrigin("gui"))
        )
        self.caretaker = PersistenceCaretaker(self.ctrl, cache_dir=cache_dir)
        self.ctrl.attach_caretaker(self.caretaker)


@pytest.fixture()
def cf(qapp, tmp_path) -> Iterator[ControllerFixture]:
    # Scope persistence to a temp dir so tests never read/write the real cache.
    fixture = ControllerFixture(cache_dir=tmp_path)
    yield fixture
    # Test producers are released before teardown; close drains their queued delivery.
    fixture.ctrl.close(timeout_ms=30_000)


def _start_interactive_analysis(cf: ControllerFixture) -> tuple[str, int]:
    tab_id = cf.ctrl.new_tab("interactive")
    cf.state.update_tab_result(tab_id, RunRecord(cfg=None, result=object()))
    token = cf.ctrl.analyze(tab_id, NoAnalyzeParams())
    cf.view.mount_interactive_analysis.assert_called_once()
    assert cf.ctrl.run_analyze_control.get_interactive(tab_id) is not None
    return tab_id, token


# ---------------------------------------------------------------------------
# project_root anchoring (default result/database paths)
# ---------------------------------------------------------------------------


def test_get_project_root_returns_injected_root(qapp, tmp_path):
    """The entry script injects the repo root; the Controller exposes it so the
    setup dialog / project RPC anchor default paths there instead of cwd (the
    .bat launcher cd's into scripts/, so cwd is the wrong base)."""
    injected = str(tmp_path / "repo_root")
    fixture = ControllerFixture(cache_dir=tmp_path, project_root=injected)
    assert fixture.ctrl.get_project_root() == injected


def test_get_project_root_falls_back_to_cwd_when_not_injected(qapp, tmp_path):
    """No injection (tests / `python -m` from the repo root) → cwd, the existing
    behaviour, so nothing regresses for callers that don't pass a root."""
    import os

    ctrl = Controller(
        state=State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry=Registry(),
        view=None,
        bus=BaseEventBus(),
    )
    assert ctrl.get_project_root() == os.getcwd()
    ctrl.close()


# ---------------------------------------------------------------------------
# new_tab / close_tab
# ---------------------------------------------------------------------------


def test_close_drains_worker_and_owner_delivery_and_closes_admission(cf):
    owner_thread = threading.get_ident()
    delivered: list[tuple[object, int]] = []
    errors: list[Exception] = []
    callback_admission: list[Exception] = []

    def on_done(value: object) -> None:
        delivered.append((value, threading.get_ident()))
        try:
            cf.ctrl.run_background(lambda: "late", delivered.append, errors.append)
        except BackgroundAdmissionClosedError as exc:
            callback_admission.append(exc)

    cf.ctrl.run_background(lambda: "result", on_done, errors.append)
    # The submitted result producer has no blocker and must deliver during close.
    cf.ctrl.close(timeout_ms=30_000)

    assert delivered == [("result", owner_thread)]
    assert errors == []
    assert len(callback_admission) == 1
    cf.ctrl.close()
    with pytest.raises(BackgroundAdmissionClosedError):
        cf.ctrl.run_background(lambda: "new", delivered.append, errors.append)
    assert delivered == [("result", owner_thread)]


def test_close_timeout_keeps_admission_closed_and_can_be_retried(cf):
    release = threading.Event()
    started = threading.Event()
    delivered: list[object] = []
    errors: list[Exception] = []

    def compute() -> str:
        started.set()
        # The test releases compute in finally; only guard a stuck caller.
        if not release.wait(30):
            raise RuntimeError("test did not release worker")
        return "finished"

    cf.ctrl.run_background(compute, delivered.append, errors.append)
    try:
        # run_background has scheduled compute, which signals entry.
        assert started.wait(30)
        with pytest.raises(TimeoutError):
            cf.ctrl.close(timeout_ms=0)
        assert delivered == []
        with pytest.raises(BackgroundAdmissionClosedError):
            cf.ctrl.run_background(lambda: "new", delivered.append, errors.append)
    finally:
        release.set()
    # compute was released in finally and will post its terminal result.
    cf.ctrl.close(timeout_ms=30_000)
    assert delivered == ["finished"]
    assert errors == []
    cf.ctrl.close()


def test_new_tab_adds_entry_to_state(cf):
    tab_id = cf.ctrl.new_tab("fake")
    assert tab_id in cf.state.tabs
    assert isinstance(cf.state.get_tab(tab_id).adapter, RecordingAdapter)


def test_new_tab_unknown_adapter_raises(cf):
    with pytest.raises(KeyError):
        cf.ctrl.new_tab("no_such_adapter")


def test_new_tab_returns_unique_ids(cf):
    t1 = cf.ctrl.new_tab("fake")
    t2 = cf.ctrl.new_tab("fake")
    assert t1 != t2


def test_close_tab_removes_from_state(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.close_tab(tab_id)
    assert tab_id not in cf.state.tabs


# ---------------------------------------------------------------------------
# cancel_analyze — agent-side teardown of an in-flight interactive analyze
# (BUG-2: without it, an interactive picker leaves is_analyzing set and the tab
# can never be closed; run.cancel can't reach the separate analyze handle).
# ---------------------------------------------------------------------------


def test_cancel_analyze_tears_down_picker_and_lets_tab_close(cf):
    tab_id, _token = _start_interactive_analysis(cf)
    assert cf.state.is_tab_analyzing(tab_id) is True
    # is_analyzing makes the tab busy, so closing it is rejected up front.
    with pytest.raises(RuntimeError, match="busy"):
        cf.ctrl.close_tab(tab_id)

    cancelled = cf.ctrl.cancel_analyze(tab_id)

    assert cancelled is True
    # The View's interactive host is torn down (dual of mount_interactive_analysis).
    cf.view.unmount_interactive_analysis.assert_called_once_with(
        tab_id, restore_result=True
    )
    # is_analyzing cleared -> the tab is no longer busy and can be closed.
    assert cf.state.is_tab_analyzing(tab_id) is False
    cf.ctrl.close_tab(tab_id)
    assert tab_id not in cf.state.tabs


def test_cancel_analyze_without_interactive_is_graceful(cf):
    tab_id = cf.ctrl.new_tab("fake")
    # No interactive analyze in flight -> graceful no-op (still unmounts defensively).
    assert cf.ctrl.cancel_analyze(tab_id) is False
    cf.view.unmount_interactive_analysis.assert_called_once_with(
        tab_id, restore_result=True
    )


# ---------------------------------------------------------------------------
# operation cancellation
# (ADR-0066): routes to the active op's OperationChannel; on stop, also
# runs op-taxonomy cancel teardown (op-taxonomy lives here, not in the View).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# cancellation capability (ADR-0066)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# start_run / run_finished flow
# ---------------------------------------------------------------------------


def test_start_run_sets_is_running(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(
        tab_id,
    )
    assert cf.state.is_tab_running(tab_id)
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))  # cleanup


def test_start_run_passes_lowered_committed_state_cfg(cf):
    """start_run delivers values accepted from the tab cfg resource."""
    tab_id = cf.ctrl.new_tab("fake")

    cfg = cf.ctrl.cfg_resources.lookup(tab_id)
    cfg.write({("reps",): 42})

    # Observe the payload at the worker-to-adapter boundary.
    captured: dict[str, RecordingCfg] = {}
    real_adapter = cf.state.get_tab(tab_id).adapter

    acquired = object()

    def _capture_run(req, accepted: RecordingCfg, *, context):
        captured["cfg"] = accepted
        assert isinstance(context.plots, Plots)
        return acquired

    spy = MagicMock(spec=RecordingAdapter)
    spy.capabilities = real_adapter.capabilities
    spy.experiment_tag = None
    spy.run.side_effect = _capture_run
    cf.state.get_tab(tab_id).adapter = spy

    cf.ctrl.start_run(
        tab_id,
    )
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))

    assert captured["cfg"].reps == 42
    assert cf.ctrl.get_tab_snapshot(tab_id).run.result is acquired


def test_start_run_emits_run_started(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(
        tab_id,
    )
    cf.bus.emit.assert_any_call(RunStartedPayload(tab_id=tab_id))
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))


def test_run_finished_updates_tab_state(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(
        tab_id,
    )
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    assert cf.state.get_tab(tab_id).run.result is not None


def test_run_finished_emits_run_finished(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(
        tab_id,
    )
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    cf.bus.emit.assert_any_call(
        RunFinishedPayload(tab_id=tab_id, outcome="finished"),
    )


def test_run_finished_calls_refresh_tab(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(
        tab_id,
    )
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    cf.bus.emit.assert_any_call(
        TabContentChangedPayload(tab_id, TabContentFact.RUN_RESULT_COMMITTED)
    )


@pytest.mark.parametrize("source", ["run", "cancelled", "load"])
def test_result_commit_survives_analysis_preparation_failure(
    cf, monkeypatch, source, tmp_path
):
    tab_id = cf.ctrl.new_tab("fake")
    adapter = cf.state.get_tab(tab_id).adapter
    monkeypatch.setattr(
        adapter,
        "capabilities",
        dataclasses.replace(adapter.capabilities, load_data=True),
    )

    def fail_params(*_args):
        raise ValueError("analysis defaults unavailable")

    loaded = SimpleNamespace(cfg_snapshot=None)
    monkeypatch.setattr(adapter, "get_analyze_params", fail_params)
    monkeypatch.setattr(adapter, "restore", lambda _stored: RestoredRun(loaded, None))
    cf.state.update_tab_analyze_param_instance(tab_id, object())
    facts = []
    cf.bus.subscribe(TabContentChangedPayload, facts.append)
    if source != "load":
        terminal = []
        cf.bus.subscribe(RunFinishedPayload, terminal.append)
        entered, release = threading.Event(), threading.Event()

        def partial_run(*_args, **_kwargs):
            entered.set()
            # Cancellation assertions release the accepted run in finally.
            if not release.wait(30):
                raise RuntimeError("test did not release partial run")
            return loaded

        if source == "cancelled":
            monkeypatch.setattr(adapter, "run", partial_run)
        cf.ctrl.start_run(tab_id)
        if source == "cancelled":
            try:
                _wait_for(entered.is_set)
                cf.ctrl.cancel_run()
            finally:
                release.set()
        _wait_for(lambda: not cf.state.is_tab_running(tab_id))
        assert terminal == [
            RunFinishedPayload(
                tab_id=tab_id,
                outcome="cancelled" if source == "cancelled" else "finished",
            )
        ]
        expected_fact = TabContentFact.RUN_RESULT_COMMITTED
        cf.view.notify_diagnostic.assert_any_call(
            "error",
            "Analysis preparation failed",
            "Run result retained. Analysis preparation failed: analysis defaults unavailable",
        )
    else:
        path = tmp_path / "loaded.hdf5"
        write_native_file(path)
        monkeypatch.setattr(adapter, "experiment_tag", "test/load")
        outcome = cf.ctrl.load_tab_result(tab_id, str(path))
        assert outcome.analysis_error == "analysis defaults unavailable"
        assert outcome.has_analyze_params is False
        expected_fact = TabContentFact.LOADED_RESULT_COMMITTED
    snapshot = cf.ctrl.get_tab_snapshot(tab_id)
    assert snapshot.run.result is not None
    if source == "load":
        assert snapshot.run.result is loaded
    assert snapshot.analysis.params is None
    assert snapshot.analysis.result is None
    assert snapshot.post_analysis.result is None
    assert facts == [TabContentChangedPayload(tab_id, expected_fact)]


def test_run_finished_skips_analyze_init_for_non_analysis_adapter(cf):
    """flux_dep / power_dep adapters declare ``analysis=AnalysisMode.NONE`` and have
    no analyze step; run-finished must not route them into analyze-params init
    (whose base impl is a Fast-Fail raise). Regression: previously this raised
    NotImplementedError and surfaced as an error dialog at the end of every run.
    """
    from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode

    tab_id = cf.ctrl.new_tab("fake")
    no_analysis = MagicMock(spec=RecordingAdapter)
    no_analysis.run.return_value = object()  # any non-NO_RESULT run result
    no_analysis.capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)
    no_analysis.experiment_tag = None
    no_analysis.get_analyze_params.side_effect = NotImplementedError(
        "declares no analysis support"
    )
    cf.state.get_tab(tab_id).adapter = no_analysis

    cf.ctrl.start_run(
        tab_id,
    )
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    # The non-analysis adapter's analyze-params builder is never touched, so the
    # run completes without surfacing an error dialog.
    no_analysis.get_analyze_params.assert_not_called()
    assert not cf.view.show_error_dialog.called
    cf.bus.emit.assert_any_call(
        TabContentChangedPayload(tab_id, TabContentFact.RUN_RESULT_COMMITTED)
    )


# ---------------------------------------------------------------------------
# run_failed flow
# ---------------------------------------------------------------------------


def test_run_failed_shows_status_message(cf):
    bad_adapter = MagicMock(spec=RecordingAdapter)
    bad_adapter.run.side_effect = RuntimeError("boom")
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    cf.ctrl.start_run(
        tab_id,
    )
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    assert cf.view.show_error_dialog.called
    msg = cf.view.show_error_dialog.call_args[0][1]
    assert "boom" in msg


def test_run_failed_clears_run_lock(cf):
    bad_adapter = MagicMock(spec=RecordingAdapter)
    bad_adapter.run.side_effect = RuntimeError("oops")
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    cf.ctrl.start_run(
        tab_id,
    )
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    assert cf.state.is_run_active() is False


# ---------------------------------------------------------------------------
# Duplicate start_run guard
# ---------------------------------------------------------------------------


def test_start_run_while_running_raises(cf):
    slow = MagicMock(spec=RecordingAdapter)
    ev = threading.Event()
    slow.run.side_effect = lambda *a, **kw: ev.wait()

    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = slow
    cf.ctrl.start_run(
        tab_id,
    )

    try:
        assert cf.state.is_tab_running(tab_id)
        other_tab_id = cf.ctrl.new_tab("fake")
        with pytest.raises(OperationConflictError, match="run is active"):
            cf.ctrl.start_run(
                other_tab_id,
            )
    finally:
        ev.set()
        _wait_for(lambda: not cf.state.is_tab_running(tab_id))


def test_start_run_while_device_setup_active_raises(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl._operation_gate.register(
        1,
        OperationKind.DEVICE_SETUP,
        owner_id="flux",
        origin_kind="user",
        note="held setup",
        resource_id="flux",
    )

    with pytest.raises(OperationConflictError, match="device_setup is active"):
        cf.ctrl.start_run(
            tab_id,
        )

    cf.ctrl._operation_gate.release(1)


def test_unselected_context_rejects_real_run(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.set_context(
        dataclasses.replace(
            cf.state.session_env,
            active_label="",
            readiness=ContextReadiness.EMPTY,
            parameters=None,
        )
    )

    with pytest.raises(RuntimeError, match="active file-backed context"):
        cf.ctrl.start_run(
            tab_id,
        )


@dataclass
class _LoadedAnalyzeParams:
    threshold: float


def test_load_tab_result_allows_unselected_context_without_soc_and_initializes_analyze(
    cf,
    tmp_path,
):
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.set_context(
        dataclasses.replace(
            cf.state.session_env,
            soc=None,
            soccfg=None,
            active_label="",
            readiness=ContextReadiness.EMPTY,
            parameters=None,
        )
    )
    loaded = object()
    adapter = MagicMock()
    # Final contract requires explicit load_data capability; RecordingAdapter's
    # default lacks it, so set load_data True for the driving path.
    from zcu_tools.gui.app.measure.adapter import AdapterCapabilities

    adapter.capabilities = AdapterCapabilities(requires_soc=False, load_data=True)
    adapter.experiment_tag = "test/load"
    adapter.restore.return_value = RestoredRun(loaded, None)
    adapter.get_analyze_params.return_value = _LoadedAnalyzeParams(threshold=0.7)
    cf.state.get_tab(tab_id).adapter = adapter
    cf.bus.emit.reset_mock()

    path = tmp_path / "loaded.hdf5"
    metadata = write_native_file(path)
    outcome = cf.ctrl.load_tab_result(tab_id, str(path))

    assert cf.state.get_tab(tab_id).run.result is loaded
    assert cf.state.get_tab(tab_id).run.source_path == str(path)
    assert cf.state.get_tab(tab_id).analysis.params == _LoadedAnalyzeParams(
        threshold=0.7
    )
    stored = adapter.restore.call_args.args[0]
    assert stored.metadata == metadata
    assert outcome.has_analyze_params is True
    cf.bus.emit.assert_any_call(
        TabContentChangedPayload(tab_id, TabContentFact.LOADED_RESULT_COMMITTED)
    )


def test_run_rejected_while_soc_connect_lease_active(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl._operation_gate.register(
        1,
        OperationKind.SOC_CONNECT,
        owner_id="soc",
        origin_kind="user",
        note="held connect",
    )

    with pytest.raises(OperationConflictError, match="soc_connect is active"):
        cf.ctrl.start_run(
            tab_id,
        )

    cf.ctrl._operation_gate.release(1)


def test_device_connect_updates_state_and_presents_ui_feedback(
    cf: ControllerFixture,
):
    """The default FakeDevice connects through the public device control facet."""
    cf.ctrl.device_control.start_connect_device(
        ConnectDeviceRequest(type_name="FakeDevice", name="flux", address="addr")
    )
    _wait_for(lambda: not cf.ctrl.device_control.get_active_device_operations())

    dev = cf.state.get_device("flux")
    assert dev is not None and dev.status is DeviceStatus.CONNECTED
    cf.view.show_status_message.assert_any_call("Device connected: flux")

    cf.ctrl.device_control.start_disconnect_device(
        DisconnectDeviceRequest(name="flux", remember=False)
    )
    _wait_for(lambda: cf.state.get_device("flux") is None)
    cf.view.show_status_message.assert_any_call("Device disconnected: flux")


def test_run_completion_prepares_pure_tab_snapshot(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(
        tab_id,
    )
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))

    snapshot = cf.ctrl.get_tab_snapshot(tab_id)

    assert snapshot.interaction.has_run_result is True
    assert snapshot.analysis is not None
    assert snapshot.analysis.params is not None


def test_cfg_edit_publishes_without_full_tab_interaction_rebuild(cf):

    tab_id = cf.ctrl.new_tab("fake")
    cfg = cf.ctrl.cfg_resources.lookup(tab_id)
    published = []
    cfg.subscribe(lambda: published.append(cfg.input))
    cf.bus.emit.reset_mock()
    cfg.write({("reps",): 42})
    assert cf.bus.emit.call_args_list == []
    assert len(published) == 1


def test_local_analyze_and_post_params_emit_precise_zero_reaction_facts(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.bus.emit.reset_mock()

    cf.ctrl.update_tab_analyze_param_instance(tab_id, object())
    cf.ctrl.update_tab_post_analyze_param_instance(tab_id, object())

    facts = [
        call.args[0].fact
        for call in cf.bus.emit.call_args_list
        if isinstance(call.args[0], TabInteractionChangedPayload)
    ]
    assert facts == [
        TabInteractionFact.ANALYZE_PARAMS_CHANGED,
        TabInteractionFact.POST_ANALYZE_PARAMS_CHANGED,
    ]


def test_tab_cfg_reset_restores_adapter_default_on_same_resource(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cfg = cf.ctrl.cfg_resources.lookup(tab_id)
    cfg.write({("reps",): 123})
    cfg.reset()
    assert cf.state.get_tab(tab_id).cfg is cfg
    assert cfg.input.data["reps"] == 100


def test_reset_tab_cfg_while_running_raises(cf):
    slow = MagicMock(spec=RecordingAdapter)
    ev = threading.Event()
    slow.run.side_effect = lambda *a, **kw: ev.wait()

    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = slow
    cf.ctrl.start_run(
        tab_id,
    )
    try:
        assert cf.state.is_tab_running(tab_id)

        from zcu_tools.gui.app.measure.services.form_contracts import (
            CfgPreconditionError,
        )

        cfg = cf.ctrl.cfg_resources.lookup(tab_id)
        with pytest.raises(CfgPreconditionError, match="blocked"):
            cfg.reset()
    finally:
        ev.set()
        _wait_for(lambda: not cf.state.is_tab_running(tab_id))


def test_get_adapter_names_includes_fake(cf):
    assert "fake" in cf.ctrl.get_adapter_names()


def test_get_adapter_names_preserves_injected_order(cf):
    assert cf.ctrl.get_adapter_names() == ["fake", "interactive"]


def test_persist_then_restore_app_state(tmp_path):
    """Full round-trip through the PersistenceCaretaker single-file memento:
    capture (flush) on one Controller, restore on a fresh one sharing the dir."""
    cf = ControllerFixture(cache_dir=tmp_path)
    tab_id = cf.ctrl.new_tab("fake")

    cfg = cf.ctrl.cfg_resources.lookup(tab_id)
    cfg.write({("reps",): 42})
    cf.ctrl.setup_control.new_entry(EntryRequest("chip"))
    cf.ctrl.setup_control.remember_connection(
        ConnectionPreferences(ip="10.0.0.2", port=7000)
    )
    cf.ctrl.persist_all()

    cf_restored = ControllerFixture(cache_dir=tmp_path)
    cf_restored.ctrl.restore_all()

    assert len(cf_restored.state.tabs) == 1
    restored_tab = next(iter(cf_restored.state.tabs.values()))
    assert restored_tab.adapter_name == "fake"
    assert restored_tab.cfg.input.data["reps"] == 42
    # remembered settings round-tripped (prefill values; project not auto-applied).
    prefs = cf_restored.ctrl.setup_control.get_setup_preferences()
    assert prefs.entry_name == "chip"
    assert prefs.ip == "10.0.0.2"
    assert prefs.port == 7000
    assert cf_restored.state.session_env.readiness is ContextReadiness.ACTIVE


def test_restore_tab_events_use_system_origin(tmp_path) -> None:
    source = ControllerFixture(cache_dir=tmp_path / "source", mock_emit=False)
    source.ctrl.new_tab("fake")
    persisted = source.ctrl.capture_persisted_state()
    restored = ControllerFixture(cache_dir=tmp_path / "restored", mock_emit=False)
    observed: list[EventMeta] = []
    from zcu_tools.gui.app.measure.events.tab import TabAddedPayload

    restored.bus.subscribe_with_meta(
        TabAddedPayload, lambda _payload, meta: observed.append(meta)
    )

    restored.ctrl.restore_persisted_state(persisted)

    assert len(observed) == 1
    assert observed[0].origin == EventOrigin(kind="system")


def test_restore_corrupt_file_is_visible_to_user(cf, tmp_path):
    """A corrupt / wrong-version state file → defaults + a user-visible error."""
    cf.caretaker._path.parent.mkdir(parents=True, exist_ok=True)
    cf.caretaker._path.write_text("{ not valid json", encoding="utf-8")

    cf.ctrl.restore_all()

    title, _ = cf.view.show_error_dialog.call_args.args
    assert title == "Settings restore failed"


def test_restore_invalid_tab_is_rejected_and_reported(cf):
    cf.ctrl._workspace_svc = MagicMock()
    cf.ctrl._workspace_svc.apply_session.return_value = RestoreReport(
        restored_tabs=0,
        rejected_tabs=(RestoreIssue("fake", "invalid saved configuration (bad cfg)"),),
    )
    cf.ctrl.restore_all()

    title, message = cf.view.show_error_dialog.call_args.args
    assert title == "Some session tabs were not restored"
    assert "invalid saved configuration" in message


# ---------------------------------------------------------------------------
# Connection / predictor — handled by ConnectionService.
# (Direct controller setters are removed in Phase 62.2.)
# ---------------------------------------------------------------------------
