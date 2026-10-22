"""Native device adapter borrowing the session's registered drivers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.device import BaseDeviceInfo, DeviceMapping
from zcu_tools.experiment.stop_signal import StopSignal

if TYPE_CHECKING:
    from zcu_tools.gui.session.services.device import DeviceService


class SessionDevices:
    """Access borrowed session drivers in the workflow worker under its RUN lease.

    No connection lifecycle, GUI cache writes, retry or clamping occurs here.
    The selected driver's model owns full cfg validation; its setup owns limits
    and I/O. Every error propagates with its original cause.
    """

    def __init__(self, service: DeviceService) -> None:
        """Borrow the session's public driver facet without lifecycle ownership."""
        self._service = service

    def get_device_cfg(self, name: str) -> BaseDeviceInfo:
        """Live-read a detached actual Info subclass; unknown names raise ValueError.

        Call in the workflow worker, not through owner-thread cache facades.
        Unknown or disconnected errors identify name. Read errors propagate.
        """
        devices = self._service.get_connected_devices()
        if name not in devices:
            raise ValueError(f"Device not connected: {name!r}")
        return devices[name].get_info().model_copy(deep=True)

    def setup_device(
        self, name: str, cfg: BaseDeviceInfo, cancel_signal: StopSignal
    ) -> None:
        """Validate complete cfg with the selected model before any driver setup.

        Revalidate dumped fields because Pydantic model_copy can bypass validators.
        Preserve all selected-model fields, including its stored units. Driver
        setup receives the original StopSignal's Event, never a replacement.
        Unknown/disconnected names raise ValueError naming name; validation,
        limits, address and I/O failures propagate without retries or fallback.
        """
        devices = self._service.get_connected_devices()
        if name not in devices:
            raise ValueError(f"Device not connected: {name!r}")
        driver = devices[name]
        validated = driver.info_model.model_validate(cfg.model_dump(mode="python"))
        driver.setup(validated, progress=False, stop_event=cancel_signal.event)

    def get_connected_devices(self) -> DeviceMapping:
        """Borrow a detached driver mapping; lifecycle stays with DeviceService.

        Read on the workflow worker under the retained RUN lease. Empty means
        no connected devices. This accessor performs no hardware I/O or writes.
        """
        return self._service.get_connected_devices()
