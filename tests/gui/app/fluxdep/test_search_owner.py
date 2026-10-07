"""Public search-owner admission, snapshot and terminal contracts."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import (
    DatabaseSearchResult,
    ParamBounds,
    SearchCancelled,
)
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload, SearchChangedPayload
from zcu_tools.gui.app.fluxdep.search import FluxDepSearchRuntime
from zcu_tools.gui.event_bus import EventOrigin
from zcu_tools.gui.expected_error import (
    ExpectedError,
    FailedPreconditionError,
    InvalidInputError,
)
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.ports import ProgressEvent
from zcu_tools.gui.session.services.progress import ProgressService


class ControlledBackground:
    """Run detached work on a foreign thread, deliver only when instructed."""

    def __init__(self) -> None:
        self.work: Callable[[], object] | None = None
        self.done: Callable[[object], None] | None = None
        self.error: Callable[[Exception], None] | None = None
        self.fail_submit = False

    def submit(
        self,
        work: Callable[[], object],
        *,
        run_in_pool: bool,
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if self.fail_submit:
            raise RuntimeError("submit failed")
        self.work, self.done, self.error = work, on_done, on_error

    def compute(self) -> object:
        assert self.work is not None
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(self.work).result()

    def deliver(self, result: object) -> None:
        assert self.done is not None
        self.done(result)

    def fail(self, exc: Exception) -> None:
        assert self.error is not None
        self.error(exc)


class QueuedProgress:
    def __init__(self, owner: ManualOwnerScheduler) -> None:
        self.owner = owner
        self.receiver: Callable[[ProgressEvent], None] | None = None

    def set_receiver(self, receiver: Callable[[ProgressEvent], None]) -> None:
        self.receiver = receiver

    def emit(self, event: ProgressEvent) -> None:
        receiver = self.receiver
        assert receiver is not None
        self.owner.post(lambda: receiver(event))


@pytest.fixture
def search_case(cross_controller, monkeypatch, tmp_path):
    from zcu_tools.gui.app.fluxdep.services import fit

    owner = ManualOwnerScheduler()
    background = ControlledBackground()
    progress = ProgressService(QueuedProgress(owner))
    ctrl = Controller(
        cross_controller.state,
        interactive_owner=owner,
        search_runtime=FluxDepSearchRuntime(background, progress),
    )
    path = tmp_path / "db.h5"
    path.touch()
    bounds = ParamBounds(EJ=(2, 15), EC=(0.2, 2), EL=(0.1, 2))
    ctrl.set_fit_params(
        str(path),
        bounds.EJ,
        bounds.EC,
        bounds.EL,
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    result = DatabaseSearchResult(
        params=(5, 1, 0.5),
        best_distance=0.1,
        best_scale=1,
        best_index=0,
        entry_results=np.array([[0.1, 1.0]]),
        entry_params=np.array([[5.0, 1.0, 0.5]]),
        fluxs=np.array([0.0]),
        freqs=np.array([4.0]),
        predicted_freqs=np.array([4.0]),
        bounds=bounds,
    )
    captured = []

    def compute(fluxs, freqs, path, transitions, bounds, *, execution):
        assert not owner.is_owner_thread()
        captured.append((fluxs.copy(), freqs.copy(), path, transitions, bounds))
        if execution.cancel_requested():
            raise SearchCancelled("requested")
        return result

    monkeypatch.setattr(fit, "search_database", compute)
    yield ctrl, background, owner, progress, result, captured
    ctrl.search.begin_close()
    ctrl.interactive.dispose()


def test_snapshot_is_captured_before_worker_and_single_flight(search_case):
    ctrl, bg, _owner, _progress, result, captured = search_case
    token = ctrl.search.start()
    assert ctrl.search.active_token == token
    with pytest.raises(RuntimeError, match="pending|busy"):
        ctrl.search.start()
    ctrl.state.spectrums["a"].points["fluxs"][0] = 0.9
    ctrl.state.fit.transitions["transitions"].append((0, 2))
    assert bg.compute() is result
    np.testing.assert_array_equal(captured[0][0], [0, 0.5, 1, 0.5])
    assert captured[0][3]["transitions"] == [(0, 1)]
    bg.deliver(result)
    assert ctrl.search.outcome(token) == OperationOutcome("finished")
    assert ctrl.search.result is result
    assert ctrl.state.fit.params == result.params


@pytest.mark.parametrize(
    "mutation", ["selection", "fit", "project", "points", "empty", "membership"]
)
def test_changed_input_rejects_success(search_case, mutation):
    ctrl, bg, _owner, _progress, result, _captured = search_case
    facts = []
    ctrl.bus.subscribe(FitChangedPayload, facts.append)
    token = ctrl.search.start()
    if mutation == "selection":
        ctrl.set_selection(np.ones(4, dtype=np.bool_))
    elif mutation == "fit":
        fit = ctrl.state.fit
        ctrl.set_fit_params(
            fit.database_path, fit.EJb, fit.ECb, fit.ELb, fit.transitions, None, None
        )
    elif mutation == "project":
        ctrl.setup_project(ctrl.state.project)
    elif mutation == "points":
        ctrl.set_points("a", np.array([0.2]), np.array([4.1]))
    elif mutation == "empty":
        ctrl.set_points("empty", np.array([]), np.array([]))
    else:
        entry = ctrl.state.spectrums["empty"]
        ctrl.remove_spectrum("empty")
        ctrl.state.put_spectrum(entry)
    bg.deliver(result)
    outcome = ctrl.search.outcome(token)
    assert outcome is not None and outcome.status == "failed"
    assert "search inputs changed" in outcome.error
    assert ctrl.state.fit.params is None
    assert not any(fact.has_result for fact in facts)
    assert ctrl.search.active_token is None


def test_active_switch_does_not_stale_and_terminal_cannot_reenter(search_case):
    ctrl, bg, _owner, _progress, result, _captured = search_case
    attempts = []

    def listener(payload):
        if payload.has_result:
            with pytest.raises(RuntimeError, match="pending|busy"):
                ctrl.search.start()
            attempts.append(ctrl.search.active_token)

    ctrl.bus.subscribe(FitChangedPayload, listener)
    token = ctrl.search.start()
    ctrl.set_active_spectrum("b")
    bg.deliver(result)
    assert attempts == [token]
    assert ctrl.search.outcome(token).status == "finished"


@pytest.mark.parametrize("terminal", ["cancelled", "failed", "late_success"])
@pytest.mark.parametrize("closing", [False, True])
def test_cancel_and_close_follow_worker_terminal(search_case, terminal, closing):
    ctrl, bg, _owner, _progress, result, _captured = search_case
    token = ctrl.search.start()
    ctrl.search.cancel(token)
    assert ctrl.search.outcome(token) is None
    if closing:
        ctrl.search.begin_close()
        with pytest.raises(RuntimeError, match="closing"):
            ctrl.search.start()
    if terminal == "cancelled":
        with pytest.raises(SearchCancelled):
            bg.compute()
        bg.fail(SearchCancelled("requested"))
        expected = "cancelled"
    elif terminal == "failed":
        bg.fail(ValueError("bad database"))
        expected = "failed"
        assert ctrl.search.outcome(token).error == "bad database"
    else:
        bg.deliver(result)
        expected = "cancelled" if closing else "finished"
    assert ctrl.search.outcome(token).status == expected
    assert ctrl.search.current.status == expected
    assert ctrl.search.active_token is None
    assert ctrl.state.fit.params == (result.params if expected == "finished" else None)
    ctrl.search.cancel(token)


def test_submit_failure_cleanup_and_known_token_queries(search_case):
    ctrl, bg, _owner, progress, _result, _captured = search_case
    bg.fail_submit = True
    with pytest.raises(RuntimeError, match="submit failed") as caught:
        ctrl.search.start()
    assert not isinstance(caught.value, ExpectedError)
    activity = ctrl.search.current
    assert activity is not None and activity.status == "failed"
    assert ctrl.search.outcome(activity.token).status == "failed"
    assert ctrl.search.active_token is None
    assert progress.bars_for_owner("fluxdep-search") == ()
    bg.fail_submit = False
    assert ctrl.search.start() != activity.token
    for query in (ctrl.search.outcome, ctrl.search.cancel):
        with pytest.raises(InvalidInputError) as caught:
            query(999999)
        assert caught.value.reason_code == "unknown_operation"
        assert isinstance(caught.value.__cause__, KeyError)
    with pytest.raises(RuntimeError, match="owner"):
        ctrl.search.await_outcome(activity.token, 0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert (
            pool.submit(ctrl.search.await_outcome, ctrl.search.active_token, 0)
            .result()
            .reason
            == "timeout"
        )
        with pytest.raises(InvalidInputError) as caught:
            pool.submit(ctrl.search.await_outcome, 999999, 0).result()
        assert caught.value.reason_code == "unknown_operation"
        assert isinstance(caught.value.__cause__, KeyError)
    assert ctrl.search.active_token is not None


def test_origin_capture_and_last_success_survives_failure(search_case):
    ctrl, bg, _owner, _progress, result, _captured = search_case
    observed = []
    ctrl.bus.subscribe_with_meta(
        SearchChangedPayload, lambda payload, meta: observed.append((payload, meta))
    )
    with ctrl.bus.origin(EventOrigin(kind="agent", client_id="client")):
        token = ctrl.search.start()
    bg.deliver(result)
    payload, meta = observed[-1]
    assert payload.token == token and payload.status == "finished"
    assert meta.origin.kind == "agent"
    assert meta.origin.client_id == "client"
    assert meta.origin.operation_id == str(token)
    next_token = ctrl.search.start()
    bg.fail(ValueError("database broke"))
    assert ctrl.search.result is result
    assert ctrl.state.fit.params == result.params
    assert ctrl.search.outcome(next_token).status == "failed"


def test_missing_background_rejects_before_handle(cross_controller):
    assert cross_controller.search.current is None
    with pytest.raises(RuntimeError, match="background|executor") as caught:
        cross_controller.search.start()
    assert not isinstance(caught.value, ExpectedError)
    assert cross_controller.search.current is None


def test_capture_and_owner_queries_reject_foreign_threads(search_case):
    ctrl, _bg, _owner, _progress, _result, _captured = search_case
    with ThreadPoolExecutor(max_workers=1) as pool:
        for query in (
            ctrl.capture_search,
            ctrl.search.start,
            lambda: ctrl.search.current,
        ):
            with pytest.raises(RuntimeError, match="owner|thread") as caught:
                pool.submit(query).result()
            assert not isinstance(caught.value, ExpectedError)
    assert ctrl.search.current is None


@pytest.mark.parametrize("closing", [False, True])
def test_search_admission_is_nominal_and_preserves_pending_handle(search_case, closing):
    ctrl, bg, _owner, _progress, _result, _captured = search_case
    token = ctrl.search.start()
    before = ctrl.state.version.snapshot()
    if closing:
        ctrl.search.begin_close()
    with pytest.raises(FailedPreconditionError) as caught:
        ctrl.search.start()
    assert caught.value.reason_code == ("search_closing" if closing else "search_busy")
    assert ctrl.search.active_token == token
    assert ctrl.search.outcome(token) is None
    assert ctrl.state.version.snapshot() == before
    bg.fail(SearchCancelled("requested"))


def test_await_on_owner_is_unexpected(search_case):
    ctrl, bg, _owner, _progress, _result, _captured = search_case
    token = ctrl.search.start()
    with pytest.raises(RuntimeError, match="owner") as caught:
        ctrl.search.await_outcome(token, 0)
    assert not isinstance(caught.value, ExpectedError)
    bg.fail(SearchCancelled("requested"))


def test_progress_notifications_cannot_reenter_start(search_case):
    ctrl, bg, _owner, progress, result, _captured = search_case
    observed = []

    def listener():
        try:
            ctrl.search.start()
        except RuntimeError as exc:
            observed.append(str(exc))

    dispose = progress.attach_by_owner("fluxdep-search", listener)
    try:
        token = ctrl.search.start()
        bg.deliver(result)
        assert len(observed) == 2
        assert all("pending" in error for error in observed)
        assert ctrl.search.outcome(token).status == "finished"
    finally:
        dispose()


def test_failed_publication_releases_admission(search_case, monkeypatch):
    from zcu_tools.gui.app.fluxdep.services.fit import FitService

    ctrl, bg, _owner, _progress, result, _captured = search_case

    def fail_record(self, result):
        raise ValueError("commit refused")

    monkeypatch.setattr(FitService, "record_result", fail_record)
    token = ctrl.search.start()
    bg.deliver(result)
    assert ctrl.search.outcome(token) == OperationOutcome("failed", "commit refused")
    assert ctrl.search.current.status == "failed"
    assert ctrl.search.active_token is None
    assert ctrl.state.fit.params is None
    assert ctrl.search.result is None
    assert ctrl.search.start() != token


@pytest.mark.parametrize("origin", ["user", "agent"])
def test_search_start_reveals_only_agent_pending_without_opening_filter(
    qapp, search_case, origin
):
    from qtpy import QtWidgets
    from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
    from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
    from zcu_tools.gui.event_bus import EventOrigin

    ctrl, bg, _owner, _progress, _result, _captured = search_case
    window = MainWindow(ctrl)
    window.show()
    try:
        stack = window.findChild(QtWidgets.QStackedWidget)
        assert stack is not None
        original = stack.currentWidget()
        assert window.findChild(AnalyzePanelWidget) is None
        with ctrl.bus.origin(EventOrigin(kind=origin)):
            token = ctrl.search.start()
        panel = window.findChild(AnalyzePanelWidget)
        if origin == "agent":
            assert panel is not None
            assert stack.currentWidget() is panel
            assert panel.current_tab == "search"
        else:
            assert panel is None
            assert stack.currentWidget() is original
        assert ctrl.interactive.inspect() is None

        ctrl.set_active_spectrum("b")
        user_view = stack.currentWidget()
        bg.fail(SearchCancelled("requested"))
        outcome = ctrl.search.outcome(token)
        assert outcome is not None and outcome.status == "cancelled"
        assert stack.currentWidget() is user_view
        assert ctrl.interactive.inspect() is None
    finally:
        if ctrl.search.active_token is not None:
            bg.fail(SearchCancelled("cleanup"))
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_search_panel_observes_app_start_progress_cancel_and_reopen(
    qapp, search_case, monkeypatch
):
    from qtpy import QtWidgets
    from zcu_tools.gui.app.fluxdep.services import fit
    from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
    from zcu_tools.progress_bar import make_pbar

    ctrl, bg, owner, _progress, result, _captured = search_case

    def compute(*args, execution):
        bar = make_pbar(total=10, desc="Database")
        bar.update(3)
        bar.close()
        return result

    monkeypatch.setattr(fit, "search_database", compute)
    panel = AnalyzePanelWidget(ctrl)
    panel.show()
    tabs = panel.findChild(QtWidgets.QTabWidget)
    assert tabs is not None
    tabs.setCurrentIndex(1)
    buttons = {
        button.text(): button for button in panel.findChildren(QtWidgets.QPushButton)
    }
    try:
        token = ctrl.search.start()
        assert not buttons["Search database"].isEnabled()
        assert buttons["Cancel search"].isEnabled()
        assert bg.compute() is result
        owner.pump_all()
        bar = panel.findChild(QtWidgets.QProgressBar)
        assert bar is not None and bar.maximum() == 10 and bar.value() == 3
        panel.detach()
        panel.hide()
        assert ctrl.search.active_token == token
        panel.activate()
        panel.show()
        assert buttons["Cancel search"].isEnabled()
        buttons["Cancel search"].click()
        assert ctrl.search.outcome(token) is None
        bg.fail(SearchCancelled("requested"))
        assert ctrl.search.outcome(token).status == "cancelled"
        assert buttons["Search database"].isEnabled()
        assert not buttons["Cancel search"].isEnabled()
        panel.dispose()
        panel.deleteLater()
        panel = AnalyzePanelWidget(ctrl)
        assert any(
            label.text() == "Search cancelled."
            for label in panel.findChildren(QtWidgets.QLabel)
        )
    finally:
        panel.quiesce()
        panel.dispose()
        panel.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("notification", ["search", "progress"])
def test_startup_notification_close_waits_for_search_terminal(
    qapp, search_case, notification
):
    from qtpy import QtWidgets
    from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
    from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
    from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner

    ctrl, bg, _owner, progress, result, _captured = search_case
    runner = BackgroundRunner()
    window = MainWindow(ctrl, search_runner=runner)
    window.show()
    close_results = []

    def close_during_startup(*_args):
        if ctrl.search.active_token is not None and not close_results:
            # No worker exists yet, so the runner can report drained during start.
            assert bg.work is None
            close_results.append(window.close())

    if notification == "search":
        dispose = ctrl.bus.subscribe(
            SearchChangedPayload, close_during_startup
        ).unsubscribe
    else:
        dispose = progress.attach_by_owner("fluxdep-search", close_during_startup)
    try:
        analyze = next(
            button
            for button in window.findChildren(QtWidgets.QPushButton)
            if button.text().startswith("Analyze")
        )
        analyze.click()
        panel = window.findChild(AnalyzePanelWidget)
        assert panel is not None
        token = ctrl.search.start()
        assert close_results == [False]
        assert window.isVisible()
        assert window.findChild(AnalyzePanelWidget) is panel
        assert ctrl.search.active_token == token
        with pytest.raises(RuntimeError, match="closing"):
            ctrl.search.start()
        bg.deliver(result)
        assert ctrl.search.active_token is None
        assert ctrl.search.outcome(token).status == "cancelled"
        assert ctrl.state.fit.params is None
        assert any(
            label.text() == "Search cancelled."
            for label in panel.findChildren(QtWidgets.QLabel)
        )
        assert window.close()
        assert not window.isVisible()
    finally:
        dispose()
        if ctrl.search.active_token is not None:
            bg.deliver(result)
        runner.quiesce()
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_close_refuses_disposal_until_search_drains(qapp, search_case):
    from qtpy import QtWidgets
    from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
    from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
    from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner

    ctrl, bg, _owner, _progress, result, _captured = search_case

    class DrainRunner(BackgroundRunner):
        drained = False

        def quiesce(self, timeout_ms: int = 5000) -> bool:
            return self.drained

    runner = DrainRunner()
    window = MainWindow(ctrl, search_runner=runner)
    window.show()
    try:
        analyze = next(
            button
            for button in window.findChildren(QtWidgets.QPushButton)
            if button.text().startswith("Analyze")
        )
        analyze.click()
        panel = window.findChild(AnalyzePanelWidget)
        assert panel is not None
        token = ctrl.search.start()
        assert not window.close()
        assert window.isVisible()
        assert window.findChild(AnalyzePanelWidget) is panel
        assert ctrl.search.active_token == token
        with pytest.raises(RuntimeError, match="closing"):
            ctrl.search.start()
        bg.deliver(result)
        assert ctrl.search.outcome(token).status == "cancelled"
        assert ctrl.state.fit.params is None
        assert any(
            label.text() == "Search cancelled."
            for label in panel.findChildren(QtWidgets.QLabel)
        )
        runner.drained = True
        assert window.close()
        assert not window.isVisible()
    finally:
        runner.drained = True
        window.close()
        window.deleteLater()
        qapp.processEvents()
