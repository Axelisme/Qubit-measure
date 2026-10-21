"""Per-case worker ownership for device service contract tests."""

from collections.abc import Iterator

import pytest

from tests.gui.session.services._device_waits import DeviceWorkers


@pytest.fixture
def device_workers(qapp) -> Iterator[DeviceWorkers]:
    """Own fresh workers until blocked drivers and queued deliveries finish."""
    workers = DeviceWorkers()
    try:
        yield workers
    finally:
        workers.close()
