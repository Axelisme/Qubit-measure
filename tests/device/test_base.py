from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, Literal, cast

import pytest
from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from zcu_tools.device import BaseDevice, BaseDeviceInfo, FakeDevice, FakeDeviceInfo


@pytest.mark.parametrize("invalid_call", [FakeDevice], ids=["FakeDevice"])
def test_fake_device_rejects_positional_fast_mode(
    invalid_call: Callable[..., object],
) -> None:
    with pytest.raises(TypeError, match="positional"):
        invalid_call(True)


class DummyDeviceInfo(BaseDeviceInfo):
    type: Literal["DummyDevice"] = "DummyDevice"


class DummySession:
    def __init__(self) -> None:
        self.resource_name = "DUMMY::INSTR"
        self.read_termination = ""
        self.write_termination = ""
        self.close_called = False

    def write(self, cmd: str) -> object:
        return None

    def query(self, cmd: str) -> str:
        if cmd == "*IDN?":
            return "dummy-idn"
        raise ValueError(f"unsupported query: {cmd}")

    def close(self) -> None:
        self.close_called = True


class DummyResourceManager:
    def __init__(self, session: DummySession) -> None:
        self.session = session
        self.opened_addresses: list[str] = []

    def open_resource(self, address: str) -> DummySession:
        self.opened_addresses.append(address)
        return self.session


class DummyDevice(BaseDevice[DummyDeviceInfo]):
    info_model = DummyDeviceInfo

    def _setup(
        self,
        cfg: DummyDeviceInfo,
        *,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None:
        return None

    def get_info(self) -> DummyDeviceInfo:
        return DummyDeviceInfo(address=self.address)


def test_subclass_without_info_model_raises_at_class_definition() -> None:
    with pytest.raises(TypeError, match="MissingInfoModel.*info_model"):

        class MissingInfoModel(BaseDevice[FakeDeviceInfo]):
            def _setup(
                self,
                cfg: FakeDeviceInfo,
                *,
                progress: bool = True,
                stop_event: threading.Event | None = None,
            ) -> None:
                return None

            def get_info(self) -> FakeDeviceInfo:
                return FakeDeviceInfo(address=self.address)


def test_subclass_with_base_info_model_raises_at_class_definition() -> None:
    with pytest.raises(TypeError, match="BaseInfoModelDevice.*concrete"):

        class BaseInfoModelDevice(BaseDevice[BaseDeviceInfo]):
            info_model = BaseDeviceInfo

            def _setup(
                self,
                cfg: BaseDeviceInfo,
                *,
                progress: bool = True,
                stop_event: threading.Event | None = None,
            ) -> None:
                return None

            def get_info(self) -> BaseDeviceInfo:
                return BaseDeviceInfo(address=self.address, type="BaseDeviceInfo")


def test_subclass_with_invalid_info_model_raises_at_class_definition() -> None:
    with pytest.raises(TypeError, match="InvalidInfoModelDevice.*BaseDeviceInfo"):

        class InvalidInfoModelDevice(BaseDevice[BaseDeviceInfo]):
            # tolerated: Invalid info_model must fail at subclass definition.
            info_model = object  # pyright: ignore[reportAssignmentType]

            def _setup(
                self,
                cfg: BaseDeviceInfo,
                *,
                progress: bool = True,
                stop_event: threading.Event | None = None,
            ) -> None:
                return None

            def get_info(self) -> BaseDeviceInfo:
                return BaseDeviceInfo(address=self.address, type="BaseDeviceInfo")


def test_base_session_methods_fail_fast_without_session() -> None:
    dev = FakeDevice(fast_mode=True)

    with pytest.raises(RuntimeError, match="has no VISA session"):
        dev.write("*RST")

    with pytest.raises(RuntimeError, match="has no VISA session"):
        dev.query("*IDN?")

    with pytest.raises(RuntimeError, match="has no VISA session"):
        dev.connect_message()


def test_connect_and_close_log_without_printing(
    caplog: LogCaptureFixture,
    capsys: CaptureFixture[str],
) -> None:
    session = DummySession()
    rm = DummyResourceManager(session)

    with caplog.at_level(logging.INFO, logger=DummyDevice.__module__):
        # tolerated(cast): Wire the recording session double through the VISA-typed constructor.
        dev = DummyDevice("DUMMY::INSTR", cast(Any, rm))
        dev.close()

    captured = capsys.readouterr()
    messages = [record.getMessage() for record in caplog.records]

    assert captured.out == ""
    assert rm.opened_addresses == ["DUMMY::INSTR"]
    assert session.close_called is True
    assert "connected to device: dummy-idn" in messages
    assert "disconnecting from DUMMY::INSTR" in messages
