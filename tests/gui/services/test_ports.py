"""Verify application services depend on ports (interfaces), not concrete infra
(docs/adr/0008 §Driven Adapter): a service runs against an in-memory fake
implementing the port, with no ExperimentManager / files involved.

(Persistence is no longer a store-port: the PersistenceCaretaker is a Driven
Adapter owning disk I/O directly, and StartupService is stateless against State —
see test_caretaker / test_startup.)
"""

from __future__ import annotations

from typing import Optional

from zcu_tools.gui.adapter import ExpContext
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.ports import ProjectIOPort
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


class _FakeProjectIO:
    """In-memory ProjectIOPort — no ExperimentManager / files."""

    def __init__(self) -> None:
        self._project = False
        self._label: Optional[str] = None

    @property
    def has_project(self) -> bool:
        return self._project

    def setup(self, result_dir: str) -> None:
        self._project = True

    def list_contexts(self) -> list[str]:
        return [] if self._label is None else [self._label]

    def get_active_label(self) -> Optional[str]:
        return self._label

    def use_context(self, label: str, base_ctx: ExpContext) -> ExpContext:
        self._label = label
        return base_ctx

    def new_context(self, base_ctx, value=None, unit="A", clone_from_current=False):
        return base_ctx


def test_concrete_io_manager_satisfies_port():
    from zcu_tools.gui.io_manager import IOManager

    assert isinstance(IOManager(), ProjectIOPort)


def test_context_service_runs_against_fake_io():
    from zcu_tools.gui.services.context import ContextService

    fake = _FakeProjectIO()
    assert isinstance(fake, ProjectIOPort)

    svc = ContextService(_make_state(), fake, EventBus())  # injected fake — no files
    assert svc.has_project() is False
    fake.setup("/tmp/whatever")
    assert svc.has_project() is True


def _make_state():
    from zcu_tools.gui.state import State

    return State(ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None))
