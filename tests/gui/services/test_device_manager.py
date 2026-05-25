# pyright: reportAttributeAccessIssue=false
"""Tests for _DeviceSetupWorker and DeviceService device operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.gui.services.device import DeviceService, _DeviceSetupWorker


def _make_svc() -> DeviceService:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import ExpContext, State

    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    return DeviceService(state, EventBus())


def test_device_service_registration():
    svc = _make_svc()
    dev = MagicMock()

    if "test_dev" in GlobalDeviceManager.get_all_devices():
        GlobalDeviceManager.drop_device("test_dev")

    svc.register_device("test_dev", dev)
    assert "test_dev" in svc.list_devices()

    svc.get_device_info("test_dev")
    dev.get_info.assert_called_once()

    svc.set_device_value("test_dev", 42)
    dev.set_value.assert_called_with(42)

    svc.get_device_value("test_dev")
    dev.get_value.assert_called_once()

    svc.drop_device("test_dev")
    assert "test_dev" not in svc.list_devices()


def test_device_setup_worker_success(qapp):
    dev = MagicMock()
    dev.setup.return_value = None

    worker = _DeviceSetupWorker(dev, "test_dev", {"param": 1}, None)

    loop = QEventLoop()
    running_at_notification = []
    worker.setup_finished.connect(
        lambda _: running_at_notification.append(worker.isRunning()) or loop.quit()
    )
    worker.failed.connect(lambda n, err: loop.quit())
    worker.cancelled.connect(lambda n: loop.quit())

    worker.start()
    loop.exec()

    dev.setup.assert_called_once()
    assert running_at_notification == [False]


def test_device_setup_worker_failure(qapp):
    dev = MagicMock()
    dev.setup.side_effect = RuntimeError("setup failed")

    worker = _DeviceSetupWorker(dev, "test_dev", {"param": 1}, None)

    loop = QEventLoop()
    error_msg = []
    running_at_notification = []

    def on_failure(_name, error):
        error_msg.append(error)
        running_at_notification.append(worker.isRunning())
        loop.quit()

    worker.failed.connect(on_failure)

    worker.start()
    loop.exec()

    dev.setup.assert_called_once()
    assert len(error_msg) == 1
    assert error_msg[0] == "setup failed"
    assert running_at_notification == [False]


def test_device_setup_worker_cancel(qapp):
    dev = MagicMock()

    def slow_setup(info, stop_event):
        stop_event.wait(0.1)

    dev.setup.side_effect = slow_setup

    worker = _DeviceSetupWorker(dev, "test_dev", {"param": 1}, None)

    loop = QEventLoop()
    was_cancelled = []
    running_at_notification = []

    def on_cancelled(_name):
        was_cancelled.append(True)
        running_at_notification.append(worker.isRunning())
        loop.quit()

    worker.cancelled.connect(on_cancelled)

    worker.start()
    worker.cancel()
    loop.exec()

    dev.setup.assert_called_once()
    assert len(was_cancelled) == 1
    assert running_at_notification == [False]


def test_device_setup_worker_pbar_factory(qapp):
    dev = MagicMock()

    def fake_factory(*args, **kwargs):
        class FakeCtx:
            def __enter__(self):
                pass

            def __exit__(self, *args):
                pass

        return FakeCtx()

    worker = _DeviceSetupWorker(dev, "test_dev", {}, fake_factory)

    loop = QEventLoop()
    worker.setup_finished.connect(loop.quit)

    worker.start()
    loop.exec()

    dev.setup.assert_called_once()


def test_device_service_setup_device(qapp):
    svc = _make_svc()
    dev = MagicMock()

    if "test_dev" in GlobalDeviceManager.get_all_devices():
        GlobalDeviceManager.drop_device("test_dev")
    GlobalDeviceManager.register_device("test_dev", dev)

    worker = svc.setup_device("test_dev", {"param": 2})
    assert isinstance(worker, _DeviceSetupWorker)

    loop = QEventLoop()
    worker.setup_finished.connect(loop.quit)
    worker.failed.connect(lambda n, err: loop.quit())
    loop.exec()

    dev.setup.assert_called_once()
    GlobalDeviceManager.drop_device("test_dev")
