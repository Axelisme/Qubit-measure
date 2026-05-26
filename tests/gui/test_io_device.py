"""Phase 5 tests — IOManager (real ExperimentManager) + DeviceManager (FakeDevice)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.device import FakeDevice, FakeDeviceInfo, GlobalDeviceManager
from zcu_tools.gui.adapter import ExpContext
from zcu_tools.gui.services.device import (
    ConnectDeviceRequest,
    DeviceService,
    DisconnectDeviceRequest,
    SetDeviceValueRequest,
)


def _make_svc(driver: object | None = None) -> tuple[DeviceService, object]:
    from zcu_tools.gui.event_bus import EventBus

    fake_device = driver if driver is not None else FakeDevice()

    def factory(type_name: str, address: str) -> object:
        return fake_device

    return DeviceService(EventBus(), driver_factory=factory), fake_device  # type: ignore[arg-type]


def _register(svc: DeviceService, name: str = "flux") -> None:
    loop = QEventLoop()
    svc.device_connected.connect(lambda _request: loop.quit())
    svc.start_connect_device(
        ConnectDeviceRequest(type_name="FakeDevice", name=name, address="")
    )
    loop.exec()


def _disconnect(svc: DeviceService, name: str = "flux") -> None:
    loop = QEventLoop()
    svc.device_disconnected.connect(lambda _request: loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name=name))
    loop.exec()


def _set_value(svc: DeviceService, name: str, value: float) -> None:
    loop = QEventLoop()
    svc.value_set.connect(lambda _name: loop.quit())
    svc.start_set_device_value(SetDeviceValueRequest(name=name, value=value))
    loop.exec()


from zcu_tools.gui.io_manager import IOManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_base_ctx(**overrides) -> ExpContext:
    defaults = dict(
        md=MagicMock(),
        ml=MagicMock(),
        soc=object(),
        soccfg=object(),
        database_path="/db",
        predictor=None,
    )
    defaults.update(overrides)
    return ExpContext(**defaults)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _clean_gdm():
    """Ensure GlobalDeviceManager is empty before and after each test."""
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)
    yield
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)


# ---------------------------------------------------------------------------
# IOManager — setup / list_contexts
# ---------------------------------------------------------------------------


def test_iomanager_not_setup_returns_empty_contexts():
    io = IOManager()
    assert io.list_contexts() == []


def test_iomanager_not_setup_raises_on_use_context():
    io = IOManager()
    with pytest.raises(RuntimeError, match="not set up"):
        io.use_context("anything", _make_base_ctx())


def test_iomanager_not_setup_returns_none_label():
    io = IOManager()
    assert io.get_active_label() is None


def test_iomanager_setup_creates_exp_dir(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))
    assert io.list_contexts() == []


# ---------------------------------------------------------------------------
# IOManager — use_context
# ---------------------------------------------------------------------------


def test_iomanager_use_context_preserves_soc_and_predictor(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))

    # create a context first
    ctx0 = _make_base_ctx()
    io.new_context(ctx0)
    label = io.get_active_label()
    assert label is not None
    label_str: str = label

    # now use_context with a different base_ctx that has soc / predictor set
    soc_obj = object()
    pred_obj = object()
    base = _make_base_ctx(soc=soc_obj, predictor=pred_obj)
    result_ctx = io.use_context(label_str, base)

    assert result_ctx.soc is soc_obj
    assert result_ctx.predictor is pred_obj
    assert result_ctx.database_path == base.database_path


def test_iomanager_use_context_updates_md_ml(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))

    ctx0 = _make_base_ctx()
    io.new_context(ctx0)
    label = io.get_active_label()
    assert label is not None

    base = _make_base_ctx()
    old_md = base.md
    old_ml = base.ml

    result_ctx = io.use_context(label, base)
    # md and ml should come from ExperimentManager, not the old mocks
    assert result_ctx.md is not old_md
    assert result_ctx.ml is not old_ml


# ---------------------------------------------------------------------------
# IOManager — new_context
# ---------------------------------------------------------------------------


def test_iomanager_new_context_creates_new_label(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))
    ctx0 = _make_base_ctx()
    io.new_context(ctx0, value=1e-3, unit="A")
    label = io.get_active_label()
    assert label is not None
    assert "mA" in label or "A" in label


def test_iomanager_new_context_result_ctx_has_fresh_md_ml(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))
    base = _make_base_ctx()
    result_ctx = io.new_context(base)
    assert result_ctx.md is not base.md
    assert result_ctx.ml is not base.ml


def test_iomanager_new_context_preserves_database_path(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))
    base = _make_base_ctx(database_path="/special/db")
    result_ctx = io.new_context(base)
    assert result_ctx.database_path == "/special/db"


def test_iomanager_new_context_clone_creates_separate_files(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))
    ctx0 = _make_base_ctx()
    ctx1 = io.new_context(ctx0)
    ctx2 = io.new_context(ctx1, clone_from_current=True)
    assert ctx1.md is not ctx2.md
    assert ctx1.ml is not ctx2.ml
    assert len(io.list_contexts()) == 2


# ---------------------------------------------------------------------------
# DeviceManager
# ---------------------------------------------------------------------------


def test_devicemanager_register_and_list(qapp):
    dev = FakeDevice()
    svc, _ = _make_svc(driver=dev)
    _register(svc, "flux")
    entries = svc.list_devices()
    entry = next((e for e in entries if e.name == "flux"), None)
    assert entry is not None
    assert entry.type_name == "FakeDevice"
    assert entry.is_connected is True


def test_devicemanager_drop_device(qapp):
    svc, _ = _make_svc()
    _register(svc, "flux")
    _disconnect(svc)
    # drop moves device to memory-only; it still appears but disconnected
    entries = svc.list_devices()
    flux_entry = next((e for e in entries if e.name == "flux"), None)
    assert flux_entry is not None
    assert not flux_entry.is_connected


def test_devicemanager_get_set_value(qapp):
    dev = FakeDevice()
    dev.set_value(3.14)
    svc, _ = _make_svc(driver=dev)
    _register(svc, "flux")

    assert svc.get_device_value("flux") == pytest.approx(3.14)
    _set_value(svc, "flux", 2.71)
    assert svc.get_device_value("flux") == pytest.approx(2.71)


def test_devicemanager_get_all_info(qapp):
    dev = FakeDevice()
    dev.set_value(1.0)
    svc, _ = _make_svc(driver=dev)
    _register(svc, "flux")
    info = GlobalDeviceManager.get_all_info()
    assert "flux" in info
    flux_info = info["flux"]
    assert isinstance(flux_info, FakeDeviceInfo)
    assert flux_info.value == pytest.approx(1.0)
