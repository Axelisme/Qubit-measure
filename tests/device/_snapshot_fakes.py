"""Connected fake driver that records checkpoint ordering and I/O failures."""

from collections.abc import Callable
from typing import Self

from zcu_tools.device import FakeDevice, FakeDeviceInfo


class SnapshotDevice(FakeDevice):
    """Real single-value driver with observable reads and experiment writes."""

    info_model = FakeDeviceInfo

    def __new__(cls) -> Self:
        """Keep this test driver's no-argument constructor anonymous."""
        return super().__new__(cls)

    def __init__(self) -> None:
        super().__init__(fast_mode=True)
        self.events: list[str] = []
        self.before_read: Callable[[], None] | None = None
        self.read_failure: OSError | None = None

    def get_info(self) -> FakeDeviceInfo:
        self.events.append("read")
        if self.before_read is not None:
            self.before_read()
        if self.read_failure is not None:
            raise self.read_failure
        return super().get_info()

    def set_value(self, value: float) -> float:
        self.events.append("write")
        return super().set_value(value)
