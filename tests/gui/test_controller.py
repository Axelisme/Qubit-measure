"""Controller skeleton (tab + run flow)."""

from __future__ import annotations

import dataclasses
import threading
import time
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication, QEventLoop
from qtpy.QtWidgets import QLabel, QStackedWidget
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    ContextReadiness,
    DirectValue,
    ExpContext,
)
from zcu_tools.gui.app.main.controller import Controller
from zcu_tools.gui.app.main.event_bus import (
    GuiEvent,
    RunFinishedPayload,
    RunStartedPayload,
    TabContentChangedPayload,
)
from zcu_tools.gui.app.main.io_manager import IOManager
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.runner import Runner
from zcu_tools.gui.app.main.services import PersistenceCaretaker, StartupProjectRequest
from zcu_tools.gui.app.main.services.device import ConnectDeviceRequest
from zcu_tools.gui.app.main.services.operation_gate import (
    OperationConflictError,
    OperationKind,
    OperationOutcome,
)
from zcu_tools.gui.app.main.services.ports import RestoreIssue, RestoreReport
from zcu_tools.gui.app.main.state import DeviceStatus, State
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.plotting.routing import has_current_container

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=MagicMock(),  # simulate connected soc
        soccfg=MagicMock(),
        res_name="fake_res",
        result_dir="/tmp/zcu_result",
        database_path="/tmp/zcu_db/fake_chip/fake_qubit",
        active_label="ctx001",
        readiness=ContextReadiness.ACTIVE,
    )


def _make_view() -> MagicMock:
    view = MagicMock()
    view.show_status_message = MagicMock()
    view.show_error_dialog = MagicMock()
    view.make_live_container = MagicMock(return_value=None)

    # The Controller fans diagnostics out via notify_diagnostic (ADR-0013);
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

    def __init__(self, cache_dir=None) -> None:
        self.state = State(_make_ctx())
        self.runner = Runner()
        self.registry = Registry()
        register_all(self.registry)
        if not self.registry.has("fake"):
            self.registry.register("fake", FakeAdapter)
        self.view = _make_view()
        io_manager = IOManager()
        io_manager._em = MagicMock()  # simulate a project being set up
        from zcu_tools.gui.app.main.event_bus import EventBus

        self.bus = EventBus()
        self.bus.emit = MagicMock()
        self.ctrl = Controller(
            state=self.state,
            runner=self.runner,
            registry=self.registry,
            io_manager=io_manager,
            view=self.view,
            bus=self.bus,
        )
        self.caretaker = PersistenceCaretaker(self.ctrl, cache_dir=cache_dir)
        self.ctrl.attach_caretaker(self.caretaker)


@pytest.fixture()
def cf(qapp, tmp_path) -> ControllerFixture:  # noqa: ARG001
    # Scope persistence to a temp dir so tests never read/write the real cache.
    return ControllerFixture(cache_dir=tmp_path)


def _default_fake_schema(ctx: ExpContext) -> CfgSchema:
    return FakeAdapter().make_default_cfg(ctx)


def _wait_for(condition, timeout_ms: int = 3000, step_ms: int = 10) -> bool:
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(step_ms / 1000)
    return False


def _make_figure_container() -> FigureContainer:
    stack = QStackedWidget()
    placeholder = QLabel("(placeholder)")
    stack.addWidget(placeholder)
    return FigureContainer(stack, placeholder)


# ---------------------------------------------------------------------------
# new_tab / close_tab
# ---------------------------------------------------------------------------


def test_new_tab_adds_entry_to_state(cf):
    tab_id = cf.ctrl.new_tab("fake")
    assert tab_id in cf.state.tabs
    assert isinstance(cf.state.get_tab(tab_id).adapter, FakeAdapter)


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
# start_run / run_finished flow
# ---------------------------------------------------------------------------


def test_start_run_sets_is_running(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id)
    assert cf.state.is_tab_running(tab_id)
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))  # cleanup


def test_start_run_uses_committed_state_schema(cf):
    """start_run reads cfg from State, not from a passed-in schema."""
    tab_id = cf.ctrl.new_tab("fake")

    # Mutate committed cfg in State after tab creation.
    base = cf.state.get_tab(tab_id).cfg_schema
    mutated_value = dataclasses.replace(
        base.value,
        fields={**base.value.fields, "reps": DirectValue(42)},
    )
    mutated = dataclasses.replace(base, value=mutated_value)
    cf.ctrl.update_tab_cfg(tab_id, mutated)

    # Intercept run to capture the schema the adapter actually receives.
    captured: dict[str, CfgSchema] = {}
    real_adapter = cf.state.get_tab(tab_id).adapter

    def _capture_run(req, schema):
        captured["schema"] = schema
        return real_adapter.run(req, schema)

    spy = MagicMock(spec=FakeAdapter)
    spy.capabilities = real_adapter.capabilities
    spy.run.side_effect = _capture_run
    cf.state.get_tab(tab_id).adapter = spy

    cf.ctrl.start_run(tab_id)
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))

    reps_value = captured["schema"].value.fields["reps"]
    assert isinstance(reps_value, DirectValue)
    assert reps_value.value == 42


def test_start_run_emits_run_started(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id)
    cf.bus.emit.assert_any_call(GuiEvent.RUN_STARTED, RunStartedPayload(tab_id=tab_id))
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))


def test_run_finished_updates_tab_state(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id)
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    assert cf.state.get_tab(tab_id).run_result is not None


def test_run_finished_emits_run_finished(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id)
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    cf.bus.emit.assert_any_call(
        GuiEvent.RUN_FINISHED,
        RunFinishedPayload(tab_id=tab_id, outcome="finished"),
    )


def test_run_finished_calls_refresh_tab(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id)
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    cf.bus.emit.assert_any_call(
        GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id=tab_id)
    )


# ---------------------------------------------------------------------------
# run_failed flow
# ---------------------------------------------------------------------------


def test_run_failed_shows_status_message(cf):
    bad_adapter = MagicMock(spec=FakeAdapter)
    bad_adapter.run.side_effect = RuntimeError("boom")
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    cf.ctrl.start_run(tab_id)
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    assert cf.view.show_error_dialog.called
    msg = cf.view.show_error_dialog.call_args[0][1]
    assert "boom" in msg


def test_run_failed_clears_run_lock(cf):
    bad_adapter = MagicMock(spec=FakeAdapter)
    bad_adapter.run.side_effect = RuntimeError("oops")
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = bad_adapter

    cf.ctrl.start_run(tab_id)
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    assert cf.state.is_run_active() is False


# ---------------------------------------------------------------------------
# Duplicate start_run guard
# ---------------------------------------------------------------------------


def test_start_run_while_running_raises(cf):
    slow = MagicMock(spec=FakeAdapter)
    ev = threading.Event()
    slow.run.side_effect = lambda *a, **kw: ev.wait()

    tab_id = cf.ctrl.new_tab("fake")
    cf.state.get_tab(tab_id).adapter = slow
    cf.ctrl.start_run(tab_id)

    assert cf.state.is_tab_running(tab_id)
    other_tab_id = cf.ctrl.new_tab("fake")
    with pytest.raises(OperationConflictError, match="run is active"):
        cf.ctrl.start_run(other_tab_id)

    # cleanup
    ev.set()
    _wait_for(lambda: not cf.state.is_tab_running(tab_id), timeout_ms=2000)


def test_start_run_while_device_setup_active_raises(cf):
    tab_id = cf.ctrl.new_tab("fake")
    lease = cf.ctrl._operation_gate.acquire(
        OperationKind.DEVICE_SETUP, owner_id="flux", resource_id="flux"
    )

    with pytest.raises(OperationConflictError, match="device_setup is active"):
        cf.ctrl.start_run(tab_id)

    cf.ctrl._operation_gate.release(lease, OperationOutcome("finished"))


def test_draft_context_rejects_real_run_and_save(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.state.set_context(
        dataclasses.replace(
            cf.state.exp_context,
            active_label="",
            readiness=ContextReadiness.DRAFT,
        )
    )

    with pytest.raises(RuntimeError, match="active file-backed context"):
        cf.ctrl.start_run(tab_id)
    with pytest.raises(RuntimeError, match="active file-backed context"):
        cf.ctrl.save_data(tab_id, "/tmp/data.h5")
    with pytest.raises(RuntimeError, match="active file-backed context"):
        cf.ctrl.save_image(tab_id, "/tmp/image.png")
    with pytest.raises(RuntimeError, match="active file-backed context"):
        cf.ctrl.save_result(tab_id, "/tmp/data.h5", "/tmp/image.png")


def test_run_rejected_while_soc_connect_lease_active(cf):
    tab_id = cf.ctrl.new_tab("fake")
    lease = cf.ctrl._operation_gate.acquire(OperationKind.SOC_CONNECT, owner_id="soc")

    with pytest.raises(OperationConflictError, match="soc_connect is active"):
        cf.ctrl.start_run(tab_id)

    cf.ctrl._operation_gate.release(lease, OperationOutcome("finished"))


def test_device_connect_handler_is_ui_only_no_persistence_coordination(cf):
    """Persistence is now a State projection (StartupService subscribes to
    DEVICE_CHANGED). The Controller's connect handler must not itself coordinate
    persistence — it only presents UI feedback."""
    driver = MagicMock()
    driver.get_info.return_value = FakeDeviceInfo(address="addr")
    cf.ctrl._dev_svc._driver_factory = lambda _type, _address: driver
    cf.ctrl._startup_svc = MagicMock()
    loop = QEventLoop()
    cf.ctrl._dev_svc.device_connected.connect(lambda _request: loop.quit())

    cf.ctrl.start_connect_device(
        ConnectDeviceRequest(type_name="FakeDevice", name="flux", address="addr")
    )
    loop.exec()

    # Controller no longer reaches into StartupService for device persistence.
    assert not cf.ctrl._startup_svc.method_calls
    # On terminal success the device is committed to State as CONNECTED.
    dev = cf.state.get_device("flux")
    assert dev is not None and dev.status is DeviceStatus.CONNECTED
    cf.view.show_status_message.assert_called()
    GlobalDeviceManager.drop_device("flux", ignore_error=True)


def test_run_clears_active_figure_container_after_finish(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.view.make_live_container.return_value = _make_figure_container()

    cf.ctrl.start_run(tab_id)

    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    assert has_current_container() is False


# ---------------------------------------------------------------------------
# View getter methods
# ---------------------------------------------------------------------------


def test_get_tab_result_returns_last_result(cf):
    import numpy as np

    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id)
    _wait_for(lambda: not cf.state.is_tab_running(tab_id))
    result = cf.ctrl.get_tab_result(tab_id)
    assert isinstance(result.data, np.ndarray)


def test_run_completion_prepares_pure_tab_snapshot(cf):
    tab_id = cf.ctrl.new_tab("fake")
    cf.ctrl.start_run(tab_id)
    assert _wait_for(lambda: not cf.state.is_tab_running(tab_id))

    snapshot = cf.ctrl.get_tab_snapshot(tab_id)

    assert snapshot.interaction.has_run_result is True
    assert snapshot.analyze_params is not None


def test_update_tab_cfg_does_not_emit_interaction_changed(cf):
    """cfg keystrokes must not trigger a full snapshot rebuild.

    update_tab_cfg writes to State but emits no TAB_INTERACTION_CHANGED;
    validity refreshes come from CfgFormWidget.validity_changed instead.
    """
    tab_id = cf.ctrl.new_tab("fake")
    base = cf.state.get_tab(tab_id).cfg_schema
    cf.bus.emit.reset_mock()

    cf.ctrl.update_tab_cfg(tab_id, base)

    for call in cf.bus.emit.call_args_list:
        event = call.args[0] if call.args else None
        assert event != GuiEvent.TAB_INTERACTION_CHANGED, (
            f"update_tab_cfg emitted TAB_INTERACTION_CHANGED unexpectedly: {call}"
        )


def test_get_adapter_names_includes_fake(cf):
    assert "fake" in cf.ctrl.get_adapter_names()


def test_new_context_with_bind_device_resolves_unit_and_value(cf):
    # bind_device -> Controller reads unit (strict whitelist) + current value
    # from the device, then hands them to ContextService. The agent never
    # supplies a raw value/unit.
    cf.ctrl._dev_svc.get_device_unit_strict = MagicMock(return_value="A")
    cf.ctrl._dev_svc.get_device_value_for_new_context = MagicMock(return_value=0.005)
    cf.ctrl._ctx_svc.new_context = MagicMock()

    cf.ctrl.new_context(bind_device="flux", clone_from="src")

    cf.ctrl._dev_svc.get_device_unit_strict.assert_called_once_with("flux")
    cf.ctrl._dev_svc.get_device_value_for_new_context.assert_called_once_with("flux")
    cf.ctrl._ctx_svc.new_context.assert_called_once_with(
        value=0.005, unit="A", clone_from="src"
    )


def test_new_context_without_bind_device_is_unbound(cf):
    # No device -> unit "none", no value, no device read.
    cf.ctrl._dev_svc.get_device_unit_strict = MagicMock()
    cf.ctrl._dev_svc.get_device_value_for_new_context = MagicMock()
    cf.ctrl._ctx_svc.new_context = MagicMock()

    cf.ctrl.new_context()

    cf.ctrl._dev_svc.get_device_unit_strict.assert_not_called()
    cf.ctrl._dev_svc.get_device_value_for_new_context.assert_not_called()
    cf.ctrl._ctx_svc.new_context.assert_called_once_with(
        value=None, unit="none", clone_from=None
    )


def test_new_context_unknown_bind_device_raises_before_creating(cf):
    # Strict whitelist failure must propagate (Fast-Fail) and never reach
    # ContextService.
    cf.ctrl._dev_svc.get_device_unit_strict = MagicMock(
        side_effect=RuntimeError("no such device")
    )
    cf.ctrl._ctx_svc.new_context = MagicMock()

    with pytest.raises(RuntimeError):
        cf.ctrl.new_context(bind_device="ghost")

    cf.ctrl._ctx_svc.new_context.assert_not_called()


def test_persist_then_restore_app_state(tmp_path):
    """Full round-trip through the PersistenceCaretaker single-file memento:
    capture (flush) on one Controller, restore on a fresh one sharing the dir."""
    cf = ControllerFixture(cache_dir=tmp_path)
    tab_id = cf.ctrl.new_tab("fake")
    schema = _default_fake_schema(cf.state.exp_context)
    cf.ctrl.update_tab_cfg(tab_id, schema)
    cf.ctrl.update_tab_save_paths(tab_id, "/tmp/a.h5", "/tmp/b.png")
    cf.ctrl.apply_startup_project(StartupProjectRequest("chip", "qub", "res", "", ""))
    cf.ctrl.persist_all()

    cf_restored = ControllerFixture(cache_dir=tmp_path)
    cf_restored.ctrl.restore_all()

    assert len(cf_restored.state.tabs) == 1
    restored_tab = next(iter(cf_restored.state.tabs.values()))
    assert restored_tab.adapter_name == "fake"
    save_paths = cf_restored.state.get_effective_save_paths(
        next(iter(cf_restored.state.tabs.keys()))
    )
    assert save_paths is not None
    assert save_paths.data_path == "/tmp/a.h5"
    assert save_paths.image_path == "/tmp/b.png"
    # startup prefs round-tripped (prefill values; project not auto-applied).
    assert cf_restored.ctrl.get_persisted_startup().chip_name == "chip"


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
