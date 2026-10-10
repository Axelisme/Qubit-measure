"""Native device adapter borrowing the session's registered drivers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.session.services.device import DeviceService


class SessionDevices:
    """Borrow session drivers for workflow worker calls under the RUN lease.

    This adapter owns no registry or connection lifecycle.
    """

    def __init__(self, service: DeviceService) -> None:
        """Borrow the session's public driver facet without lifecycle ownership."""
        self._service = service
