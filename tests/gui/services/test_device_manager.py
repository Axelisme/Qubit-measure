# pyright: reportAttributeAccessIssue=false
"""Tests for DeviceManager and _DeviceSetupWorker."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.gui.device_manager import DeviceManager, _DeviceSetupWorker


def test_device_manager_registration():
    dm = DeviceManager()
    dev = MagicMock()

    # Clean up before
    if "test_dev" in GlobalDeviceManager.get_all_devices():
        GlobalDeviceManager.drop_device("test_dev")

    dm.register_device("test_dev", dev)
    devices = dm.list_devices()
    assert "test_dev" in devices

    # get_device_info
    dm.get_device_info("test_dev")
    dev.get_info.assert_called_once()

    # get/set value
    dm.set_device_value("test_dev", 42)
    dev.set_value.assert_called_with(42)

    dm.get_device_value("test_dev")
    dev.get_value.assert_called_once()

    # get_all_info
    info = dm.get_all_info()
    assert "test_dev" in info

    # drop device
    dm.drop_device("test_dev")
    assert "test_dev" not in dm.list_devices()


def test_device_setup_worker_success(qapp):
    dev = MagicMock()
    # dev.setup runs instantly
    dev.setup.return_value = None

    worker = _DeviceSetupWorker(dev, "test_dev", {"param": 1}, None)

    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    worker.failed.connect(lambda n, err: loop.quit())
    worker.cancelled.connect(lambda n: loop.quit())

    worker.start()
    loop.exec()

    dev.setup.assert_called_once()


def test_device_setup_worker_failure(qapp):
    dev = MagicMock()
    dev.setup.side_effect = RuntimeError("setup failed")

    worker = _DeviceSetupWorker(dev, "test_dev", {"param": 1}, None)

    loop = QEventLoop()
    error_msg = []
    worker.failed.connect(lambda n, err: error_msg.append(err) or loop.quit())

    worker.start()
    loop.exec()

    dev.setup.assert_called_once()
    assert len(error_msg) == 1
    assert error_msg[0] == "setup failed"


def test_device_setup_worker_cancel(qapp):
    dev = MagicMock()

    def slow_setup(info, stop_event):
        # wait a bit, checking stop_event
        stop_event.wait(0.1)

    dev.setup.side_effect = slow_setup

    worker = _DeviceSetupWorker(dev, "test_dev", {"param": 1}, None)

    loop = QEventLoop()
    was_cancelled = []
    worker.cancelled.connect(lambda n: was_cancelled.append(True) or loop.quit())

    worker.start()
    worker.cancel()
    loop.exec()

    dev.setup.assert_called_once()
    assert len(was_cancelled) == 1


def test_device_setup_worker_pbar_factory(qapp):
    dev = MagicMock()
    MagicMock()

    def fake_factory(*args, **kwargs):
        class FakeCtx:
            def __enter__(self):
                pass

            def __exit__(self, *args):
                pass

        return FakeCtx()

    worker = _DeviceSetupWorker(dev, "test_dev", {}, fake_factory)

    loop = QEventLoop()
    worker.finished.connect(loop.quit)

    worker.start()
    loop.exec()

    dev.setup.assert_called_once()


def test_device_manager_setup_device(qapp):
    dm = DeviceManager()
    dev = MagicMock()

    if "test_dev" in GlobalDeviceManager.get_all_devices():
        GlobalDeviceManager.drop_device("test_dev")
    dm.register_device("test_dev", dev)

    worker = dm.setup_device("test_dev", {"param": 2})
    assert isinstance(worker, _DeviceSetupWorker)

    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    worker.failed.connect(lambda n, err: loop.quit())
    loop.exec()

    dev.setup.assert_called_once()
    dm.drop_device("test_dev")
