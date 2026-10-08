"""Native device adapter borrowing the session's registered drivers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.stop_signal import StopSignal

if TYPE_CHECKING:
    from zcu_tools.gui.session.services.device import DeviceService


class SessionDevices:
    """Forward native workflow device effects through the owning session.

    This adapter borrows DeviceService; it owns no drivers or registry and never
    connects, disconnects, clamps or retries. Calls run on the workflow worker
    under its shared RUN lease. Device drivers own validation and hard limits.
    """

    def __init__(self, service: DeviceService) -> None:
        """Borrow the session's public driver facet without lifecycle ownership."""
        raise NotImplementedError

    def set_value(self, name: str, value: float, cancel_signal: StopSignal) -> None:
        """Apply absolute flux in native units with cooperative cancellation.

        Read the selected registered driver's native info, use its set_flux
        setter, and apply through the existing experiment setup_devices owner.
        Unknown/disconnected names, unsupported flux, limits and I/O raise with
        the native cause. Do not use the Start-time cached snapshot as live info.
        """
        raise NotImplementedError

    def setup(
        self, settings: Mapping[str, DeviceInfo], cancel_signal: StopSignal
    ) -> None:
        """Delegate ordered native settings to existing experiment setup_devices.

        Reuse the session's registered drivers and native name/address/model
        validation. Pass the same StopSignal; cancellation returns normally.
        This adapter does not expand the batch setup policy slated for removal.
        """
        raise NotImplementedError
