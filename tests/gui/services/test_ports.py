"""Verify application services depend on ports (interfaces), not concrete infra
(docs/adr/0008 §Driven Adapter): a service runs against an in-memory fake
implementing the port, with no ExperimentManager / files involved.

(Persistence is no longer a store-port: the PersistenceCaretaker is a Driven
Adapter owning disk I/O directly, and StartupService is stateless against State —
see test_caretaker / test_startup.)
"""

from __future__ import annotations

from zcu_tools.gui.app.main.adapter import ExpContext
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.ports import ProjectIOPort
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


class _FakeProjectIO:
    """In-memory ProjectIOPort — no ExperimentManager / files."""

    def __init__(self) -> None:
        self._project = False
        self._label: str | None = None

    @property
    def has_project(self) -> bool:
        return self._project

    def setup(self, result_dir: str) -> None:
        self._project = True

    def list_contexts(self) -> list[str]:
        return [] if self._label is None else [self._label]

    def get_active_label(self) -> str | None:
        return self._label

    def use_context(self, label: str, base_ctx: ExpContext) -> ExpContext:
        self._label = label
        return base_ctx

    def new_context(self, base_ctx, value=None, unit="none", clone_from=None):
        return base_ctx


def test_concrete_io_manager_satisfies_port():
    from zcu_tools.gui.session.services.io_manager import IOManager

    assert isinstance(IOManager(), ProjectIOPort)


def test_context_service_runs_against_fake_io():
    from zcu_tools.gui.session.services.context import ContextService

    fake = _FakeProjectIO()
    assert isinstance(fake, ProjectIOPort)

    svc = ContextService(_make_state(), fake, EventBus())  # injected fake — no files
    assert svc.has_project() is False
    fake.setup("/tmp/whatever")
    assert svc.has_project() is True


def test_writeback_service_satisfies_writeback_lifecycle_port():
    """``WritebackService`` structurally satisfies ``WritebackLifecyclePort``."""
    from unittest.mock import MagicMock

    from zcu_tools.gui.app.main.services.ports import WritebackLifecyclePort
    from zcu_tools.gui.app.main.services.writeback import WritebackService

    svc = WritebackService(
        state=_make_state(),
        cfg_editor=MagicMock(),
        write_port=MagicMock(),
    )
    assert isinstance(svc, WritebackLifecyclePort)


def test_cfg_editor_service_satisfies_cfg_editor_port():
    """``CfgEditorService`` structurally satisfies ``CfgEditorPort``."""
    from unittest.mock import MagicMock

    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorService
    from zcu_tools.gui.app.main.services.ports import CfgEditorPort

    svc = CfgEditorService(
        env_ctrl=MagicMock(),
        read_port=MagicMock(),
        write_port=MagicMock(),
        version_bump=MagicMock(),
        version_drop=MagicMock(),
        bus=EventBus(),
    )
    assert isinstance(svc, CfgEditorPort)


def _make_state():
    from zcu_tools.gui.app.main.state import State

    return State(ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None))
