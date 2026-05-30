"""M1 — verify application services depend on ports (interfaces), not concrete
infrastructure: each service runs against an in-memory fake implementing the
port, with no disk / ExperimentManager / hardware involved.

This is the payoff of the port boundary (docs/adr/0008 §Driven Adapter, M1):
the concrete ``StartupPersistenceService`` / ``SessionPersistenceService`` /
``IOManager`` satisfy the ports structurally, and a fake can be swapped in.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

from zcu_tools.gui.adapter import ExpContext
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.ports import (
    ProjectIOPort,
    SessionStorePort,
    StartupStorePort,
)
from zcu_tools.gui.services.startup_persistence import (
    STARTUP_VERSION,
    PersistedDeviceEntry,
    PersistedStartup,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

# --- concrete services satisfy their ports (structural) -------------------


def test_concrete_persistence_satisfies_ports():
    from zcu_tools.gui.services.session_persistence import SessionPersistenceService
    from zcu_tools.gui.services.startup_persistence import StartupPersistenceService

    assert isinstance(StartupPersistenceService(), StartupStorePort)
    assert isinstance(SessionPersistenceService(), SessionStorePort)


def test_concrete_io_manager_satisfies_port():
    from zcu_tools.gui.io_manager import IOManager

    assert isinstance(IOManager(), ProjectIOPort)


# --- in-memory fakes let services run without concrete infra --------------


class _FakeStartupStore:
    """In-memory StartupStorePort — no disk."""

    def __init__(self) -> None:
        self._data = PersistedStartup(
            version=STARTUP_VERSION,
            chip_name="",
            qub_name="",
            res_name="",
            result_dir="",
            database_path="",
            ip="host",
            port=8887,
            devices=[],
        )

    def load(self) -> Optional[PersistedStartup]:
        return self._data

    def get_current(self) -> PersistedStartup:
        return self._data

    def update_project(self, **kw) -> None:
        self._data = PersistedStartup(**{**self._data.__dict__, **kw})

    def update_connection(self, *, ip: str, port: int) -> None:
        self._data = PersistedStartup(**{**self._data.__dict__, "ip": ip, "port": port})

    def replace_devices(self, entries: list[PersistedDeviceEntry]) -> None:
        self._data = PersistedStartup(
            **{**self._data.__dict__, "devices": list(entries)}
        )

    def update_left_panel_width(self, width: int) -> None:
        self._data = PersistedStartup(
            **{**self._data.__dict__, "left_panel_width": width}
        )


def test_startup_service_runs_against_fake_store():
    from zcu_tools.gui.services.startup import StartupConnectionRequest, StartupService

    assert isinstance(_FakeStartupStore(), StartupStorePort)

    fake = _FakeStartupStore()
    state = _make_state()
    svc = StartupService(
        context=MagicMock(),
        devices=MagicMock(),
        persistence=fake,  # injected fake — no disk
        state=state,
        bus=EventBus(),
    )

    svc.remember_connection(StartupConnectionRequest(ip="1.2.3.4", port=9000))

    assert fake.get_current().ip == "1.2.3.4"
    assert fake.get_current().port == 9000


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
