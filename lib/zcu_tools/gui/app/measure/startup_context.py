"""Synchronous script facets over the session's existing owner commands.

Scripts run off-owner. These facets marshal each command to the owner and return
only after it commits. On-close callbacks run on-owner and use the same facets
without queuing a blocking call back to themselves. Commands preserve their
owner's validation, gates and exceptions; no script-level rollback is provided.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from zcu_tools.device.base import BaseDevice, BaseDeviceInfo
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.app.measure.services.tab_control import TabControlPort
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import (
    EntryInfo,
    EntryRequest,
    PointUseRequest,
    SavedPoint,
)
from zcu_tools.gui.session.ports import OwnerScheduler
from zcu_tools.gui.session.services.connection import SoCConnectionService
from zcu_tools.gui.session.services.device import DeviceService
from zcu_tools.gui.session.setup_control import SetupControlPort
from zcu_tools.gui.session.types import SocCfgHandle, SocHandle


@dataclass(frozen=True)
class OwnerCalls:
    """Dispatch script commands synchronously using scheduler, the session owner.

    This concentrates the worker/owner distinction shared by all script facets.
    No timeout or retry is added; the application must pump the owner while
    draining workers. Owner exceptions propagate unchanged to the caller.
    """

    scheduler: OwnerScheduler

    def call[T](self, command: Callable[[], T]) -> T:
        """Run command inline on-owner, or block a foreign caller until it returns."""
        if self.scheduler.is_owner_thread():
            return command()
        return self.scheduler.call(command)


@dataclass(frozen=True)
class StartupSoc:
    """Board installation facet; calls marshals to owner and service owns publication.

    The script keeps ownership of external board resources, including cleanup.
    No universal SoC close method is assumed.
    """

    calls: OwnerCalls
    service: SoCConnectionService

    def install(self, soc: SocHandle, soccfg: SocCfgHandle) -> None:
        """Publish caller-owned board/config through SOC_CONNECT gate.

        Replace an idle installed pair; Run/connect conflicts propagate without
        changing the current pair. Safe from script worker or owner callback.
        """
        self.calls.call(lambda: self.service.install_soc(soc, soccfg))


@dataclass(frozen=True)
class StartupDevices:
    """Driver registration facet; calls selects owner and service takes ownership."""

    calls: OwnerCalls
    service: DeviceService

    def register[Info: BaseDeviceInfo](
        self, name: str, driver: BaseDevice[Info]
    ) -> None:
        """Register a built driver and cache its info on-owner, synchronously.

        GUI owns driver close only after success. Duplicate live name, Run,
        same-device mutation and info-read failures propagate; a rejected driver
        remains caller-owned. Safe from worker or owner callback.
        """
        self.calls.call(lambda: self.service.register_device(name, driver))


@dataclass(frozen=True)
class StartupSetup:
    """Entry/point facet; calls marshals to the existing setup control owner."""

    calls: OwnerCalls
    control: SetupControlPort

    def open_entry(self, name: str) -> EntryInfo:
        """Open an existing entry by single-segment name; do not select a point.

        Owner validates name and loads entry aliases. I/O/binding failures
        propagate and preserve the old binding. Safe from worker/owner callback.
        """
        return self.calls.call(lambda: self.control.open_entry(EntryRequest(name)))

    def use_point(self, label: str) -> SavedPoint:
        """Select an existing label in the opened entry on-owner.

        Missing entry/point and resource errors propagate, preserving the old
        selection. No point is created. Safe from worker/owner callback.
        """
        return self.calls.call(lambda: self.control.use_point(PointUseRequest(label)))


@dataclass(frozen=True)
class StartupParameters:
    """Alias/component commands; calls selects owner and control owns transactions."""

    calls: OwnerCalls
    control: ContextControlPort

    def set_alias_rows(self, rows: tuple[AliasRow, ...]) -> None:
        """Replace ordered authored aliases in memory, without saving aliases.yaml.

        Invalid rows propagate without changing the old table. Entry-owned
        disk aliases remain governed by Inspect Apply. Worker/owner safe.
        """
        self.calls.call(lambda: self.control.set_alias_rows(rows))

    def add_component(
        self, name: str, kind: str, fields: YamlMap, *, setup: bool = False
    ) -> None:
        """Add registered kind with working-unit fields to point or setup template.

        setup=True targets the opened entry template, otherwise the selected
        point. The owner validates and commits one transaction. Schema, context,
        conflict and I/O errors propagate without rollback of earlier commands.
        Caller must not mutate fields until return. Worker/owner safe.
        """
        self.calls.call(
            lambda: self.control.add_component(name, kind, fields, setup=setup)
        )


@dataclass(frozen=True)
class StartupTabs:
    """Tab creation facet; calls selects owner and control owns tab lifetime."""

    calls: OwnerCalls
    control: TabControlPort

    def new_tab(self, adapter: str) -> str:
        """Create adapter's tab, capture ready sources once, and return its ID.

        adapter is a registered catalog name. Unknown adapters and form/source
        failures propagate before publication. Earlier tabs remain. Worker/owner
        safe; the GUI owns the created tab and its captured input.
        """
        return self.calls.call(lambda: self.control.new_tab(adapter))


@dataclass(frozen=True)
class StartupContext:
    """Capabilities passed to trusted startup(ctx), without persistent session state.

    soc installs caller-owned boards. devices transfers successfully registered
    drivers to GUI ownership. setup selects existing entry/point. context edits
    aliases/components. tabs creates tabs from ready sources. register_close
    records callbacks in the runner; report_progress publishes script progress.
    Both callbacks are owner commands, reached through calls. Facets and these
    methods support worker scripts and owner-thread on-close callbacks.
    """

    soc: StartupSoc
    devices: StartupDevices
    setup: StartupSetup
    context: StartupParameters
    tabs: StartupTabs
    calls: OwnerCalls
    register_close: Callable[[Callable[[], None]], None]
    report_progress: Callable[[str], None]

    def on_close(self, callback: Callable[[], None]) -> None:
        """Register callback for reverse-order, once-only execution after drain.

        callback takes no arguments and runs on the owner thread before managed
        driver close. It may use synchronous ctx facets, but must not block on
        owner delivery or submit new background work (admission is closed).
        Exceptions are logged and remaining callbacks still run. Registration
        after callbacks begin raises RuntimeError. Force kill gives no guarantee.
        """
        self.calls.call(lambda: self.register_close(callback))

    def progress(self, message: str) -> None:
        """Show nonblank progress text while startup runs, without blocking Qt.

        Empty text raises ValueError. No file or state cache is written. Safe
        from worker; reporting after startup terminal raises RuntimeError.
        """
        self.calls.call(lambda: self.report_progress(message))
