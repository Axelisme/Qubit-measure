"""Verify application services depend on ports (interfaces), not concrete infra
(docs/adr/0008 §Driven Adapter): a service runs against an in-memory fake
implementing the port, with no files involved.

(Persistence is no longer a store-port: the PersistenceCaretaker is a Driven
Adapter owning disk I/O directly, and ProjectSettingsService is stateless against State.
See tests/gui/app/measure/services/test_caretaker.py and
tests/gui/session/services/test_project_settings.py.)
"""

from __future__ import annotations

from zcu_tools.gui.event_bus import BaseEventBus as EventBus


def test_writeback_service_satisfies_writeback_lifecycle_port():
    """``WritebackService`` structurally satisfies ``WritebackLifecyclePort``."""
    from unittest.mock import MagicMock

    from zcu_tools.gui.app.measure.services.ports import WritebackLifecyclePort
    from zcu_tools.gui.app.measure.services.writeback import WritebackService

    svc = WritebackService(
        cfg_editor=MagicMock(), write_port=MagicMock(), entry_ledger=MagicMock()
    )
    assert isinstance(svc, WritebackLifecyclePort)


def test_cfg_editor_service_satisfies_cfg_editor_port():
    """``CfgEditorService`` structurally satisfies ``CfgEditorPort``."""
    from unittest.mock import MagicMock

    from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
    from zcu_tools.gui.app.measure.services.ports import CfgEditorPort

    svc = CfgEditorService(
        env_ctrl=MagicMock(),
        read_port=MagicMock(),
        write_port=MagicMock(),
        versions=MagicMock(),
        bus=EventBus(),
    )
    assert isinstance(svc, CfgEditorPort)
