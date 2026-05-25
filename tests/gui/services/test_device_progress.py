from __future__ import annotations

from qtpy.QtCore import QCoreApplication
from zcu_tools.gui.services.device_progress import (
    DeviceSetupProgressFactory,
    DeviceSetupProgressModel,
)


def test_device_setup_progress_model_retains_progress_without_widget(qapp):
    model = DeviceSetupProgressModel()
    factory = DeviceSetupProgressFactory(model)
    pbar = factory(desc="Ramp value", total=2.0, leave=False)
    QCoreApplication.processEvents()

    pbar.update(1.0)
    QCoreApplication.processEvents()

    snapshot = model.snapshot()
    assert len(snapshot) == 1
    assert snapshot[0].maximum == 10000
    assert snapshot[0].value == 5000
    assert "Ramp value" in snapshot[0].format

    pbar.close()
    QCoreApplication.processEvents()
    assert model.snapshot() == ()
