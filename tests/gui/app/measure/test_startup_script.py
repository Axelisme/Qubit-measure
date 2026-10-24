"""Startup scripts cross public owner facets and retain successful work on failure."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest
from qtpy.QtWidgets import QPushButton
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.gui import launcher
from zcu_tools.gui.app.measure import default_startup
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.ports import OperationConflictError
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.resources.entry import ResultEntry, component_registry

from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._components import QubitSchema
from tests.gui.app.measure._reload_fakes import Loader
from tests.gui.app.measure._synthetic import SampleAdapter
from tests.resources.entry.fakes import registry_state, restore_registry


class ClosingDevice(FakeDevice):
    info_model = FakeDeviceInfo

    calls: list[str]

    def close(self) -> None:
        self.calls.append("hardware")
        super().close()


class StartupProbe(ModuleType):
    calls: list[str]
    started: threading.Event
    release: threading.Event
    worker_ids: list[int]
    driver: ClosingDevice
    soccfg: object
    tab: str
    connection_threads: list[int]


@pytest.fixture
def startup_app(qapp, tmp_path, monkeypatch):
    registry = Registry()
    registry.register("sample", SampleAdapter)
    previous_hook = sys.excepthook
    behavior = MeasureGuiBehavior(
        lambda: (registry, Loader()), clean=True, project_root=str(tmp_path)
    )
    assembly = behavior.assemble(None)
    behavior.before_show(assembly)
    ctrl = assembly.controller
    window = assembly.window
    assert isinstance(ctrl, Controller) and isinstance(window, MainWindow)
    probe = StartupProbe("startup_probe")
    probe.calls = []
    probe.started = threading.Event()
    probe.release = threading.Event()
    probe.worker_ids = []
    probe.connection_threads = []
    probe.driver = ClosingDevice(fast_mode=True)
    probe.driver.calls = probe.calls
    probe.soccfg = make_mock_soccfg()
    monkeypatch.setitem(sys.modules, "startup_probe", probe)
    ctrl.setup_control.new_entry(EntryRequest("synthetic"))
    ctrl.setup_control.new_point(PointCreateRequest("first", "", CommandOrigin("gui")))
    try:
        yield behavior, assembly, ctrl, window, probe
    finally:
        probe.release.set()
        behavior.close(assembly)
        sys.excepthook = previous_hook
        qapp.processEvents()


def write_script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "startup.py"
    path.write_text(
        "import startup_probe as p\nimport threading\n\ndef startup(ctx):\n" + body,
        encoding="utf-8",
    )
    return path


def run_script(ctrl: Controller, path: Path) -> None:
    """Run a synthetic script and surface its public assertions at the runner seam."""
    token = ctrl.start_startup(path)
    qt_until(
        lambda: ctrl.operation_control.await_operation(token, 0).reason == "completed",
        label="synthetic facet script",
    )
    outcome = ctrl.operation_control.await_operation(token, 0).outcome
    assert outcome is not None and outcome.status == "finished", outcome


def test_framework_default_script_is_selectable_and_defines_startup():
    default = Path(default_startup.__file__)
    assert default.is_file()
    assert callable(default_startup.startup)
    assert (
        launcher.select_startup_script(clean=False, script=None, default_script=default)
        == default
    )


def test_script_selection_uses_only_default_custom_or_clean(tmp_path):
    default = tmp_path / "default.py"
    custom = tmp_path / "custom.py"
    assert (
        launcher.select_startup_script(clean=False, script=None, default_script=default)
        == default
    )
    assert (
        launcher.select_startup_script(
            clean=False, script=custom, default_script=default
        )
        == custom
    )
    assert (
        launcher.select_startup_script(clean=True, script=None, default_script=default)
        is None
    )
    with pytest.raises(ValueError, match="--clean.*--script"):
        launcher.select_startup_script(
            clean=True, script=custom, default_script=default
        )


def test_composition_rejects_clean_script_before_factory_and_global_hooks(tmp_path):
    factory_calls: list[str] = []

    def registry_factory():
        factory_calls.append("called")
        return Registry(), Loader()

    previous_hook = sys.excepthook
    previous_thread_hook = threading.excepthook
    try:
        with pytest.raises(ValueError, match="--clean.*--script"):
            MeasureGuiBehavior(
                registry_factory, clean=True, startup_script=tmp_path / "startup.py"
            )
        assert (factory_calls, sys.excepthook, threading.excepthook) == (
            [],
            previous_hook,
            previous_thread_hook,
        )
    finally:
        sys.excepthook = previous_hook
        threading.excepthook = previous_thread_hook


def test_background_facets_lock_run_and_keep_completed_work(startup_app, tmp_path):
    _behavior, _assembly, ctrl, _window, probe = startup_app
    path = write_script(
        tmp_path,
        "    p.worker_ids.append(threading.get_ident())\n    ctx.soc.install(p.soccfg, p.soccfg)\n    ctx.devices.register('synthetic', p.driver)\n    ctx.setup.open_entry('synthetic')\n    ctx.setup.use_point('first')\n    ctx.context.set_alias_rows(())\n    p.tab = ctx.tabs.new_tab('sample')\n    ctx.progress('Waiting for synthetic source')\n    p.started.set()\n    p.release.wait()\n",
    )
    token = ctrl.start_startup(path)
    try:
        qt_until(probe.started.is_set, label="startup owner operations")
        assert probe.worker_ids != [threading.get_ident()]
        assert ctrl.has_soc()
        assert (
            ctrl.device_control.get_device_info("synthetic") == probe.driver.get_info()
        )
        assert ctrl.tab_control.get_tab_snapshot(probe.tab).interaction.startup_active
        with pytest.raises(OperationConflictError, match="startup"):
            ctrl.run_analyze_control.start_run(probe.tab)
    finally:
        probe.release.set()
    qt_until(
        lambda: ctrl.operation_control.await_operation(token, 0).reason == "completed",
        label="startup terminal",
    )
    assert not ctrl.tab_control.get_tab_snapshot(probe.tab).interaction.startup_active
    assert ctrl.tab_control.has_tab(probe.tab)


def test_partial_failure_reports_script_line_and_retains_callbacks(
    startup_app, tmp_path, caplog
):
    behavior, assembly, ctrl, window, probe = startup_app
    window.destroyed.connect(lambda: probe.calls.append("view released"))
    path = write_script(
        tmp_path,
        "    ctx.devices.register('synthetic', p.driver)\n    ctx.on_close(lambda: p.calls.append('first'))\n    def broken():\n        p.calls.append('broken')\n        raise ValueError('callback failure')\n    ctx.on_close(broken)\n    ctx.on_close(lambda: p.calls.append('last'))\n    p.tab = ctx.tabs.new_tab('sample')\n    raise ValueError('script failure')\n",
    )
    token = ctrl.start_startup(path)
    qt_until(
        lambda: ctrl.operation_control.await_operation(token, 0).reason == "completed",
        label="failed startup",
    )
    outcome = ctrl.operation_control.await_operation(token, 0).outcome
    assert outcome is not None and outcome.status == "failed"
    assert outcome.error is not None
    assert str(path) in outcome.error and "line 13" in outcome.error
    assert "script failure" in outcome.error
    assert ctrl.tab_control.has_tab(probe.tab)
    behavior.close(assembly)
    behavior.close(assembly)
    assert probe.calls == ["last", "broken", "first", "hardware", "view released"]
    assert "callback failure" in caplog.text


def test_close_drains_startup_owner_delivery_before_callbacks_and_hardware(
    startup_app, tmp_path
):
    behavior, assembly, ctrl, window, probe = startup_app
    window.destroyed.connect(lambda: probe.calls.append("view released"))
    path = write_script(
        tmp_path,
        "    ctx.devices.register('synthetic', p.driver)\n    ctx.on_close(lambda: (ctx.context.set_alias_rows(()), p.calls.append('callback')))\n    p.started.set()\n    p.release.wait()\n    ctx.context.set_alias_rows(())\n    p.calls.append('worker finished')\n",
    )
    ctrl.start_startup(path)
    qt_until(probe.started.is_set, label="startup waiting")
    with pytest.raises(TimeoutError):
        ctrl.close(timeout_ms=0)
    assert probe.calls == []
    probe.release.set()
    behavior.close(assembly)
    assert probe.calls == ["worker finished", "callback", "hardware", "view released"]


def test_startup_creates_point_then_persists_its_aliases(startup_app, tmp_path):
    _behavior, _assembly, ctrl, _window, _probe = startup_app
    before = registry_state()
    try:
        component_registry.register("startup/qubit", QubitSchema)
        path = write_script(
            tmp_path,
            "    from zcu_tools.gui.session.aliases import AliasRow\n"
            "    ctx.setup.new_entry('created')\n"
            "    ctx.context.add_component('Q1', 'startup/qubit', {'freq': 5.25}, setup=True)\n"
            "    ctx.setup.new_point('initial')\n"
            "    ctx.context.apply_alias_rows((AliasRow('qubit', 'Q1'),))\n",
        )
        token = ctrl.start_startup(path)
        qt_until(
            lambda: (
                ctrl.operation_control.await_operation(token, 0).reason == "completed"
            ),
            label="entry creation startup",
        )
        outcome = ctrl.operation_control.await_operation(token, 0).outcome
        assert outcome is not None and outcome.status == "finished"
        assert ctrl.context_control.current_point_label() == "initial"
        assert ctrl.context_control.read_context().value("$qubit.freq") == 5.25
        entry = ResultEntry.open(
            "created",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        assert entry.load_aliases(point="initial").aliases == {"qubit": "Q1"}
        ctrl.setup_control.open_entry(EntryRequest("synthetic"))
        ctrl.setup_control.open_entry(EntryRequest("created"))
        ctrl.setup_control.use_point(PointUseRequest("initial"))
        assert ctrl.context_control.read_context().value("$qubit.freq") == 5.25
    finally:
        restore_registry(before)
        assert registry_state() == before


def test_component_and_alias_facets_commit_through_context_owner(startup_app, tmp_path):
    _behavior, _assembly, ctrl, _window, _probe = startup_app
    before = registry_state()
    try:
        component_registry.register("startup/qubit", QubitSchema)
        path = write_script(
            tmp_path,
            "    from zcu_tools.gui.session.aliases import AliasRow\n"
            "    ctx.context.add_component('Q1', 'startup/qubit', {'freq': 5.25})\n"
            "    ctx.context.set_alias_rows((AliasRow('qubit', 'Q1'),))\n",
        )
        token = ctrl.start_startup(path)
        qt_until(
            lambda: (
                ctrl.operation_control.await_operation(token, 0).reason == "completed"
            ),
            label="component startup",
        )
        outcome = ctrl.operation_control.await_operation(token, 0).outcome
        assert outcome is not None and outcome.status == "finished"
        assert ctrl.context_control.read_context().value("$qubit.freq") == 5.25
        assert "Q1" in ctrl.context_control.inspect_document().components
    finally:
        restore_registry(before)
        assert registry_state() == before


def test_live_queries_and_close_callback_read_current_not_startup_state(
    startup_app, tmp_path
):
    behavior, assembly, ctrl, _window, probe = startup_app
    path = write_script(
        tmp_path,
        "    from zcu_tools.gui.session.state import DeviceStatus\n"
        "    assert ctx.soc.current() is None\n"
        "    assert ctx.setup.current().entry.name == 'synthetic'\n"
        "    assert ctx.setup.current().point_label == 'first'\n"
        "    assert ctx.tabs.current() == ()\n"
        "    ctx.devices.register('live', p.driver)\n"
        "    row, = ctx.devices.current()\n"
        "    assert (row.name, row.type_name, row.address) == ('live', 'FakeDevice', p.driver.get_info().address)\n"
        "    assert row.status is DeviceStatus.CONNECTED\n"
        "    def closing():\n"
        "        assert ctx.setup.current().point_label == 'later'\n"
        "        assert len(ctx.tabs.current()) == 1\n"
        "        assert ctx.devices.current()[0].status is DeviceStatus.CONNECTED\n"
        "        p.calls.append('snapshot')\n"
        "    ctx.on_close(closing)\n",
    )
    run_script(ctrl, path)
    ctrl.setup_control.new_point(PointCreateRequest("later", "", CommandOrigin("gui")))
    ctrl.tab_control.new_tab("sample")
    behavior.close(assembly)
    assert probe.calls == ["snapshot", "hardware"]


def test_remember_and_connect_non_manual_device_through_known_adapter(
    startup_app, tmp_path
):
    _behavior, _assembly, ctrl, _window, _probe = startup_app
    run_script(
        ctrl,
        write_script(
            tmp_path,
            "    from zcu_tools.gui.session.state import DeviceStatus\n"
            "    ctx.devices.remember('offline', 'FakeDevice', 'startup-offline')\n"
            "    row, = ctx.devices.current()\n"
            "    assert (row.name, row.type_name, row.address) == ('offline', 'FakeDevice', 'startup-offline')\n"
            "    assert row.status is DeviceStatus.MEMORY_ONLY\n"
            "    ctx.devices.connect('offline', 'FakeDevice', 'startup-offline')\n"
            "    assert ctx.devices.current()[0].status is DeviceStatus.CONNECTED\n"
            "    try:\n"
            "        ctx.devices.remember('offline', 'FakeDevice', 'replacement')\n"
            "    except RuntimeError:\n"
            "        pass\n"
            "    else:\n"
            "        raise AssertionError('remember must reject a live identity')\n"
            "    assert ctx.devices.current()[0].address == 'startup-offline'\n",
        ),
    )
    assert ctrl.device_control.get_device_info("offline") is not None


@pytest.mark.parametrize("fails", [False, True])
def test_connection_facets_wait_off_owner_and_publish_only_success(
    startup_app, tmp_path, monkeypatch, fails
):
    from zcu_tools import qick_remote

    _behavior, _assembly, ctrl, _window, probe = startup_app
    owner_id = threading.get_ident()

    def connect(ip: str, port: int):
        assert (ip, port) == ("offline-board", 1234)
        probe.connection_threads.append(threading.get_ident())
        if fails:
            raise OSError("offline connection refused")
        return probe.soccfg, probe.soccfg

    monkeypatch.setattr(qick_remote, "make_soc_proxy", connect)
    run_script(
        ctrl,
        write_script(
            tmp_path,
            "    try:\n"
            "        ctx.soc.connect('offline-board', 1234)\n"
            "    except RuntimeError as error:\n"
            "        assert 'offline connection refused' in str(error)\n"
            "        assert ctx.soc.current() is None\n"
            "        p.calls.append('failed')\n"
            "    else:\n"
            "        current = ctx.soc.current()\n"
            "        assert (current.ip, current.port) == ('offline-board', 1234)\n"
            "        ctx.soc.install(p.soccfg, p.soccfg)\n"
            "        assert ctx.soc.current() is None\n"
            "        p.calls.append('connected')\n",
        ),
    )
    assert probe.calls == ["failed" if fails else "connected"]
    assert len(probe.connection_threads) == 1
    assert probe.connection_threads[0] != owner_id


def test_failed_device_connection_propagates_and_retains_memory_row(
    startup_app, tmp_path
):
    _behavior, _assembly, ctrl, _window, _probe = startup_app
    run_script(
        ctrl,
        write_script(
            tmp_path,
            "    from zcu_tools.gui.session.state import DeviceStatus\n"
            "    try:\n"
            "        ctx.devices.connect('unknown', 'NoSuchDriver', 'unavailable')\n"
            "    except RuntimeError as error:\n"
            "        assert 'NoSuchDriver' in str(error)\n"
            "    else:\n"
            "        raise AssertionError('unknown driver must not connect')\n"
            "    row, = ctx.devices.current()\n"
            "    assert row.status is DeviceStatus.MEMORY_ONLY and row.error\n",
        ),
    )
    assert ctrl.device_control.get_device_snapshot("unknown").error


def test_handled_script_connection_failure_uses_summary_without_dialog(
    startup_app, tmp_path
):
    _behavior, _assembly, ctrl, window, _probe = startup_app
    window.show()
    run_script(
        ctrl,
        write_script(
            tmp_path,
            "    try:\n"
            "        ctx.devices.connect('offline', 'NoSuchDriver', 'none')\n"
            "    except RuntimeError:\n"
            "        ctx.devices.remember('offline', 'NoSuchDriver', 'none')\n"
            "    ctx.summary('Startup: device failed; kept offline')\n",
        ),
    )
    ui = ComponentDriver(window)
    assert ui.dialogs() == ()
    assert ui.status().text == "Startup: device failed; kept offline"


def test_manual_device_failure_still_shows_dialog(startup_app):
    from zcu_tools.gui.session.services.device import ConnectDeviceRequest

    _behavior, _assembly, ctrl, window, _probe = startup_app
    window.show()
    token = ctrl.device_control.start_connect_device(
        ConnectDeviceRequest("NoSuchDriver", "manual", "none")
    )
    qt_until(
        lambda: ctrl.operation_control.await_operation(token, 0).reason == "completed",
        label="manual device failure",
    )
    ui = ComponentDriver(window)
    assert any(
        dialog.title == "Device operation failed: manual" for dialog in ui.dialogs()
    )
    ui.click("OK")


def test_summary_survives_terminal_and_cache_root_is_available(startup_app, tmp_path):
    _behavior, _assembly, ctrl, window, _probe = startup_app
    window.show()
    run_script(
        ctrl,
        write_script(
            tmp_path,
            "    assert ctx.cache_root.is_absolute()\n"
            "    try:\n"
            "        ctx.summary('  ')\n"
            "    except ValueError:\n"
            "        pass\n"
            "    else:\n"
            "        raise AssertionError('blank summary accepted')\n"
            "    ctx.summary('Restored 2 items; 1 failed')\n",
        ),
    )
    assert ComponentDriver(window).status().text == "Restored 2 items; 1 failed"


def test_tab_input_roundtrip_preserves_expression_selection_and_invalid_draft(
    startup_app, tmp_path
):
    _behavior, _assembly, ctrl, _window, _probe = startup_app
    original = ctrl.tab_control.new_tab("sample")
    resource = ctrl.cfg_resources.lookup(original)
    resource.write({("gain",): {"expr": "unfinished +"}, ("reps",): "not-an-int"})
    resource.select(("sweep",), {"kind": "center"})
    expected = resource.input.to_json()
    run_script(
        ctrl,
        write_script(
            tmp_path,
            "    tab, = ctx.tabs.current()\n"
            "    assert tab.adapter_name == 'sample'\n"
            "    p.tab = ctx.tabs.new_tab(tab.adapter_name, input=tab.input)\n"
            "    assert ctx.tabs.current()[1].input == tab.input\n",
        ),
    )
    restored = ctrl.cfg_resources.lookup(ctrl.tab_control.list_tab_ids()[1])
    assert restored.input.to_json() == expected
    assert restored.preview().errors


def test_new_tab_rejects_bad_input_and_logs_incompatible_paths(
    startup_app, tmp_path, caplog
):
    _behavior, _assembly, ctrl, _window, _probe = startup_app
    run_script(
        ctrl,
        write_script(
            tmp_path,
            "    from zcu_tools.gui.app.measure.services.tab_input import TabInput\n"
            "    for adapter, raw, expected in [('missing', '{}', KeyError), ('sample', 'not-json', ValueError), ('sample', '{}', ValueError)]:\n"
            "        try:\n"
            "            ctx.tabs.new_tab(adapter, input=TabInput(raw))\n"
            "        except expected:\n"
            "            pass\n"
            "        else:\n"
            "            raise AssertionError('bad tab admitted')\n"
            "    assert ctx.tabs.current() == ()\n"
            '    ctx.tabs.new_tab(\'sample\', input=TabInput(\'{"data":{"gain":0.35,"removed":1},"selections":[]}\'))\n',
        ),
    )
    assert (
        ctrl.cfg_resources.lookup(ctrl.tab_control.list_tab_ids()[0]).input.data["gain"]
        == 0.35
    )
    assert "discarded" in caplog.text and "removed" in caplog.text


def test_blocking_connections_fast_fail_from_owner_callback(startup_app, tmp_path):
    behavior, assembly, ctrl, _window, probe = startup_app
    run_script(
        ctrl,
        write_script(
            tmp_path,
            "    def closing():\n"
            "        for connect in [lambda: ctx.soc.connect('never', 1234), lambda: ctx.devices.connect('never', 'FakeDevice', 'never')]:\n"
            "            try:\n"
            "                connect()\n"
            "            except RuntimeError as error:\n"
            "                assert 'worker' in str(error)\n"
            "                p.calls.append('rejected')\n"
            "            else:\n"
            "                raise AssertionError('owner connection admitted')\n"
            "        assert ctx.soc.current() is None\n"
            "        assert ctx.devices.current() == ()\n"
            "    ctx.on_close(closing)\n",
        ),
    )
    behavior.close(assembly)
    assert probe.calls == ["rejected", "rejected"]


@pytest.mark.parametrize("selection", ["default", "custom", "clean"])
def test_composition_starts_only_selected_synthetic_script(
    startup_app, tmp_path, qapp, selection
):
    _old_behavior, _old_assembly, _ctrl, _window, probe = startup_app
    default = tmp_path / "default.py"
    custom = tmp_path / "custom.py"
    for path, label in ((default, "default"), (custom, "custom")):
        path.write_text(
            f"import startup_probe as p\ndef startup(ctx):\n    p.calls.append('{label}')\n    p.started.set()\n",
            encoding="utf-8",
        )
    selected = launcher.select_startup_script(
        clean=selection == "clean",
        script=custom if selection == "custom" else None,
        default_script=default,
    )
    behavior = MeasureGuiBehavior(
        lambda: (Registry(), Loader()),
        clean=selection == "clean",
        startup_script=selected,
        project_root=str(tmp_path),
    )
    assembly = behavior.assemble(None)
    assert isinstance(assembly.window, MainWindow)
    window = assembly.window
    try:
        window.show()
        behavior.after_show(assembly)
        delivered = threading.Event()
        if not isinstance(assembly.controller, Controller):
            raise AssertionError("composition needs a Controller")
        assembly.controller.owner_scheduler.post(delivered.set)
        qt_until(delivered.is_set, label="after-show owner turn")
        if selection == "clean":
            assert probe.calls == []
        else:
            qt_until(probe.started.is_set, label="selected script")
            driver = ComponentDriver(window)
            qt_until(
                lambda: driver.status().text == "Startup complete",
                label="startup UI terminal",
            )
            assert probe.calls == [selection]
    finally:
        behavior.close(assembly)
        qapp.processEvents()


def test_programmatic_shutdown_runs_callbacks_before_driver_close(
    startup_app, tmp_path
):
    _behavior, _assembly, ctrl, window, probe = startup_app
    path = write_script(
        tmp_path,
        "    ctx.devices.register('synthetic', p.driver)\n"
        "    ctx.on_close(lambda: p.calls.append('callback'))\n",
    )
    token = ctrl.start_startup(path)
    qt_until(
        lambda: ctrl.operation_control.await_operation(token, 0).reason == "completed",
        label="shutdown startup",
    )
    window.show()
    window.request_shutdown()
    qt_until(lambda: not window.isVisible(), label="programmatic app shutdown")
    assert probe.calls == ["callback", "hardware"]


@pytest.mark.parametrize("fails", [False, True], ids=["complete", "partial-failure"])
def test_startup_progress_and_run_lock_with_user_input(
    startup_app, tmp_path, qapp, fails
):
    _behavior, _assembly, ctrl, window, probe = startup_app
    # Composition and the external synthetic source barrier are fixture setup.
    # Clicking the real Refresh button makes that source ready, independently
    # of the startup runner's own admission/progress logic.
    soccfg = make_mock_soccfg()
    ctrl.session_services.soc_connection.install_soc(soccfg, soccfg)
    ctrl.tab_control.new_tab("sample")
    window.show()
    refresh = next(
        button
        for button in window.findChildren(QPushButton)
        if button.text() == "Refresh"
    )
    refresh.clicked.connect(probe.release.set)
    path = write_script(
        tmp_path,
        "    ctx.progress('Waiting for synthetic source')\n"
        "    p.started.set()\n"
        "    p.release.wait()\n"
        + ("    raise ValueError('synthetic partial failure')\n" if fails else ""),
    )
    ctrl.start_startup(path)
    qt_until(probe.started.is_set, label="UI startup progress")
    driver = ComponentDriver(window)
    evidence = Path(os.environ.get("STARTUP_UI_EVIDENCE", str(tmp_path))) / (
        "partial-failure" if fails else "complete"
    )
    initial_size = driver.window_size()
    assert driver.status().text == "Waiting for synthetic source"
    assert not driver.observe("Run").enabled
    assert driver.observe("Run").tooltip == "Startup script is running"
    driver.screenshot(evidence / "01-progress.png")
    driver.click("Run")
    assert not driver.observe("Run").enabled
    driver.screenshot(evidence / "02-run-blocked.png")
    driver.click("Refresh")
    if fails:
        qt_until(
            lambda: any(
                dialog.title == "Startup script failed" for dialog in driver.dialogs()
            ),
            label="startup error dialog",
        )
        dialog = next(
            dialog
            for dialog in driver.dialogs()
            if dialog.title == "Startup script failed"
        )
        assert any(
            str(path) in text
            and "line 8" in text
            and "synthetic partial failure" in text
            for text in dialog.texts
        )
        driver.screenshot(evidence / "03-error.png")
        driver.click("OK")
        assert "Startup failed:" in driver.status().text
        assert path.name in driver.status().text
    else:
        qt_until(
            lambda: driver.status().text == "Startup complete",
            label="startup completed UI",
        )
    assert driver.observe("Run").enabled
    driver.screenshot(evidence / "04-unlocked.png")
    assert driver.window_size() == initial_size, (
        "terminal status must not expand the window"
    )
