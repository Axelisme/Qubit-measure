"""Deterministic injected ports shared by synthetic async workflow tests."""

from collections.abc import Callable
from datetime import UTC, datetime
from io import StringIO

from zcu_tools.datafile import JsonObject
from zcu_tools.device import BaseDeviceInfo, FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.acquisition import capture_device_snapshot
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar

START = datetime(2026, 10, 7, tzinfo=UTC)


class ManualClock:
    """Advance only when the Engine requests a future wait."""

    def __init__(self) -> None:
        self.value = START
        self.targets: list[datetime] = []
        self.on_wait: Callable[[StopSignal], object] | None = None

    def now(self) -> datetime:
        return self.value

    def wait_until(self, target: datetime, cancel_signal: StopSignal) -> None:
        self.targets.append(target)
        if self.on_wait is not None:
            self.on_wait(cancel_signal)
        if not cancel_signal.is_set():
            self.value = target


class RecordingDevices:
    """Record requested native setpoints without touching hardware."""

    def __init__(self) -> None:
        self.values: list[tuple[str, float]] = []
        self.on_set: Callable[[StopSignal], object] | None = None
        self.configs: list[BaseDeviceInfo] = []
        self.signals: list[StopSignal] = []
        self.drivers: dict[str, FakeDevice] = {}
        self.captures = 0

    def capture_snapshot(self) -> dict[str, JsonObject]:
        self.captures += 1
        return capture_device_snapshot(self.drivers)

    def get_device_cfg(self, name: str) -> BaseDeviceInfo:
        if name in self.drivers:
            return self.drivers[name].get_info().model_copy(deep=True)
        return FakeDeviceInfo(address="memory")

    def setup_device(
        self, name: str, cfg: BaseDeviceInfo, cancel_signal: StopSignal
    ) -> None:
        value = cfg.model_dump()["value"]
        self.values.append((name, value))
        self.configs.append(cfg.model_copy(deep=True))
        self.signals.append(cancel_signal)
        if self.on_set is not None:
            self.on_set(cancel_signal)
        if name in self.drivers:
            driver = self.drivers[name]
            validated = driver.info_model.model_validate(cfg.model_dump(mode="python"))
            driver.setup(validated, progress=False, stop_event=cancel_signal.event)


class RecordingBar(TQDMProgressBar):
    """Silent runtime progress bar with observable close lifecycle."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs, file=StringIO())
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()
