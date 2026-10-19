"""Startup scripts cross public owner facets and retain successful work on failure."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.gui import launcher
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.context_types import CommandOrigin, EntryRequest, PointCreateRequest
from zcu_tools.gui.session.ports import OperationConflictError
from zcu_tools.program.v2.mocksoc import make_mock_soccfg

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._reload_fakes import Loader
from tests.gui.app.measure._synthetic import SampleAdapter


class ClosingDevice(FakeDevice):
    info_model = FakeDeviceInfo

    calls: list[str]

    def close(self) -> None:
        self.calls.append("hardware")
        super().close()


@pytest.fixture
def startup_app(qapp, tmp_path, monkeypatch):
    registry = Registry()
    registry.register("sample", SampleAdapter)
    previous_hook = sys.excepthook
    behavior = MeasureGuiBehavior(lambda: (registry, Loader()), clean=True, project_root=str(tmp_path))
    assembly = behavior.assemble(None)
    ctrl = assembly.controller
    window = assembly.window
    assert isinstance(ctrl, Controller) and isinstance(window, MainWindow)
    probe = ModuleType("startup_probe")
    probe.calls = []
    probe.started = threading.Event()
    probe.release = threading.Event()
    probe.worker_ids = []
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
        window.deleteLater()
        sys.excepthook = previous_hook
        qapp.processEvents()


def write_script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "startup.py"
    path.write_text("import startup_probe as p\nimport threading\n\ndef startup(ctx):\n" + body, encoding="utf-8")
    return path


def test_script_selection_uses_only_default_custom_or_clean(tmp_path):
    default = tmp_path / "default.py"
    custom = tmp_path / "custom.py"
    assert launcher.select_startup_script(clean=False, script=None, default_script=default) == default
    assert launcher.select_startup_script(clean=False, script=custom, default_script=default) == custom
    assert launcher.select_startup_script(clean=True, script=None, default_script=default) is None
    with pytest.raises(ValueError, match="--clean.*--script"):
        launcher.select_startup_script(clean=True, script=custom, default_script=default)


def test_background_facets_lock_run_and_keep_completed_work(startup_app, tmp_path):
    _behavior, _assembly, ctrl, _window, probe = startup_app
    path = write_script(tmp_path, "    p.worker_ids.append(threading.get_ident())\n    ctx.soc.install(p.soccfg, p.soccfg)\n    ctx.devices.register('synthetic', p.driver)\n    ctx.setup.open_entry('synthetic')\n    ctx.setup.use_point('first')\n    ctx.context.set_alias_rows(())\n    p.tab = ctx.tabs.new_tab('sample')\n    ctx.progress('Waiting for synthetic source')\n    p.started.set()\n    p.release.wait()\n")
    token = ctrl.start_startup(path)
    try:
        qt_until(probe.started.is_set, label="startup owner operations")
        assert probe.worker_ids != [threading.get_ident()]
        assert ctrl.has_soc()
        assert ctrl.device_control.get_device_info("synthetic") == probe.driver.get_info()
        assert ctrl.get_tab_snapshot(probe.tab).interaction.startup_active
        with pytest.raises(OperationConflictError, match="startup"):
            ctrl.start_run(probe.tab)
    finally:
        probe.release.set()
    qt_until(lambda: ctrl.operation_control.await_operation(token, 0).reason == "completed", label="startup terminal")
    assert not ctrl.get_tab_snapshot(probe.tab).interaction.startup_active
    assert ctrl.has_tab(probe.tab)


def test_partial_failure_reports_script_line_and_retains_callbacks(startup_app, tmp_path, caplog):
    behavior, assembly, ctrl, _window, probe = startup_app
    path = write_script(tmp_path, "    ctx.devices.register('synthetic', p.driver)\n    ctx.on_close(lambda: p.calls.append('first'))\n    def broken():\n        p.calls.append('broken')\n        raise ValueError('callback failure')\n    ctx.on_close(broken)\n    ctx.on_close(lambda: p.calls.append('last'))\n    p.tab = ctx.tabs.new_tab('sample')\n    raise ValueError('script failure')\n")
    token = ctrl.start_startup(path)
    qt_until(lambda: ctrl.operation_control.await_operation(token, 0).reason == "completed", label="failed startup")
    outcome = ctrl.operation_control.await_operation(token, 0).outcome
    assert outcome is not None and outcome.status == "failed"
    assert outcome.error is not None
    assert str(path) in outcome.error and "line 13" in outcome.error
    assert "script failure" in outcome.error
    assert ctrl.has_tab(probe.tab)
    behavior.close(assembly)
    behavior.close(assembly)
    assert probe.calls == ["last", "broken", "first", "hardware"]
    assert "callback failure" in caplog.text


def test_close_drains_startup_owner_delivery_before_callbacks_and_hardware(startup_app, tmp_path):
    behavior, assembly, ctrl, _window, probe = startup_app
    path = write_script(tmp_path, "    ctx.devices.register('synthetic', p.driver)\n    ctx.on_close(lambda: (ctx.context.set_alias_rows(()), p.calls.append('callback')))\n    p.started.set()\n    p.release.wait()\n    ctx.context.set_alias_rows(())\n    p.calls.append('worker finished')\n")
    ctrl.start_startup(path)
    qt_until(probe.started.is_set, label="startup waiting")
    with pytest.raises(TimeoutError):
        ctrl.close(timeout_ms=0)
    assert probe.calls == []
    probe.release.set()
    behavior.close(assembly)
    assert probe.calls == ["worker finished", "callback", "hardware"]
