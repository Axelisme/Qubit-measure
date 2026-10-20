"""Synchronous script facets over the session's existing owner commands.

Scripts run off-owner. These facets marshal each command to the owner and return
only after it commits. On-close callbacks run on-owner and use the same facets
without queuing a blocking call back to themselves. Commands preserve their
owner's validation, gates and exceptions; no script-level rollback is provided.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from zcu_tools.device.base import BaseDevice, BaseDeviceInfo
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.app.measure.events.startup import STARTUP_ORIGIN
from zcu_tools.gui.app.measure.services.tab_control import TabControlPort
from zcu_tools.gui.app.measure.services.tab_input import StartupTab, TabInput
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryInfo,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
    SavedPoint,
)
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.ports import OwnerScheduler
from zcu_tools.gui.session.services.connection import (
    ConnectRemoteRequest,
    SoCConnectionService,
)
from zcu_tools.gui.session.services.device import DeviceService, DeviceSnapshot
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
    bus: BaseEventBus

    def call[T](self, command: Callable[[], T]) -> T:
        """Run inline on-owner or block a worker, with script origin inside owner scope.

        bus is the session event bus. Its origin scope is created in the owner
        callback, since ContextVar does not cross worker threads. Existing
        operations capture this identity for their own terminal facts; no fact,
        failure or handle is suppressed here. Command exceptions propagate.
        """

        def on_owner() -> T:
            with self.bus.origin(STARTUP_ORIGIN):
                return command()

        if self.scheduler.is_owner_thread():
            return on_owner()
        return self.scheduler.call(on_owner)


def _wait_for_connection(handles: OperationHandles, token: int) -> None:
    # Only workers wait; owner remains free to commit the existing async terminal.
    while True:
        result = handles.await_known_outcome(token, 60)
        if result.reason == "timeout":
            continue
        outcome = result.outcome
        if outcome is None or outcome.status != "finished":
            raise RuntimeError(
                outcome.error if outcome is not None else "Missing connection outcome"
            )
        return


@dataclass(frozen=True)
class StartupSelection:
    """Current opened entry and selected point, not remembered preferences.

    entry is cached owner metadata or None when unopened. point_label is the
    selected saved label or None when no point is selected. No disk I/O occurs.
    """

    entry: EntryInfo | None
    point_label: str | None


@dataclass(frozen=True)
class StartupSoc:
    """Board installation facet; calls marshals to owner and service owns publication.

    The script keeps ownership of external board resources, including cleanup.
    No universal SoC close method is assumed. handles is the session's existing
    thread-safe operation registry used only for worker connection waits.
    """

    calls: OwnerCalls
    service: SoCConnectionService
    handles: OperationHandles

    def current(self) -> ConnectRemoteRequest | None:
        """Read the current remote endpoint on-owner without network I/O.

        Worker/owner safe. None means absent or caller-installed without an
        endpoint. This detached projection does not confer ownership of a board.
        """
        return self.calls.call(self.service.current_request)

    def connect(self, address: str, port: int) -> None:
        """Connect address/TCP port via the owner's async SOC_CONNECT operation.

        Worker-only: owner calls raise RuntimeError before admission. Blocks
        this worker, not Qt, until terminal; no timeout/retry is added. Gate or
        submission errors propagate. Async failures raise RuntimeError with the
        service diagnostic and preserve the previous pair. GUI owns publication,
        not board cleanup; scripts must not assume a universal close contract.
        """
        if self.calls.scheduler.is_owner_thread():
            raise RuntimeError("Blocking connect requires a script worker")
        token = self.calls.call(
            lambda: self.service.start_connect(ConnectRemoteRequest(address, port))
        )
        _wait_for_connection(self.handles, token)

    def install(self, soc: SocHandle, soccfg: SocCfgHandle) -> None:
        """Publish caller-owned board/config through SOC_CONNECT gate.

        Replace an idle installed pair; Run/connect conflicts propagate without
        changing the current pair. Safe from script worker or owner callback.
        """
        self.calls.call(lambda: self.service.install_soc(soc, soccfg))


@dataclass(frozen=True)
class StartupDevices:
    """Device facet; calls selects owner, service owns drivers and row publication.

    handles is the existing session registry used for off-owner terminal waits.
    """

    calls: OwnerCalls
    service: DeviceService
    handles: OperationHandles

    def current(self) -> tuple[DeviceSnapshot, ...]:
        """Read detached live and memory-only rows on-owner, without hardware I/O.

        Worker/owner safe. Name/type/address/status/error describe current state,
        not connectivity guarantees. Caller owns the returned tuple.
        """
        return self.calls.call(self.service.list_device_snapshots)

    def remember(self, name: str, type_name: str, address: str) -> None:
        """Publish a memory-only name/type/address through the device owner.

        Worker/owner safe; no disk or hardware I/O and no driver ownership.
        Blank fields, Run, live/busy name conflicts propagate without mutation.
        Known adapters can Reconnect even when manual GUI Add is disabled;
        unknown types have no factory. Identical idle rows retain their error.
        """
        self.calls.call(lambda: self.service.remember_device(name, type_name, address))

    def connect(self, name: str, type_name: str, address: str) -> None:
        """Remember and Reconnect a trusted identity through the existing owner.

        Worker-only; owner calls raise RuntimeError before creating a row/work.
        Blocks this worker until async terminal, never Qt. No timeout/retry.
        Admission errors propagate; driver failures raise RuntimeError with the
        diagnostic and retain the offline row/error. GUI owns successful driver
        close. Unknown types never gain an implicit factory.
        """
        if self.calls.scheduler.is_owner_thread():
            raise RuntimeError("Blocking connect requires a script worker")

        def start() -> int:
            self.service.remember_device(name, type_name, address)
            return self.service.start_reconnect_device(name)

        token = self.calls.call(start)
        _wait_for_connection(self.handles, token)

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

    def current(self) -> StartupSelection:
        """Read entry/point in one owner turn, without loading any source.

        Worker/owner safe. Return detached metadata; caller owns the value,
        while the existing context owner retains entry/point resource handles.
        """
        return self.calls.call(
            lambda: StartupSelection(
                self.control.get_entry_info(), self.control.current_point_label()
            )
        )

    def new_entry(self, name: str) -> EntryInfo:
        """Create and open an empty entry named by one safe path segment.

        Use the GUI's data roots; select no point. Existing destinations and
        invalid names fail without overwriting. I/O errors propagate; an
        installation failure retains the created entry for open_entry recovery.
        The owner manages files/binding, with no cross-file rollback.
        Synchronous from a worker or owner callback.
        """
        return self.calls.call(lambda: self.control.new_entry(EntryRequest(name)))

    def new_point(self, label: str) -> SavedPoint:
        """Create and select label in the opened entry, seeded from its setup.

        label is one safe path segment. Use an empty description/library and GUI
        attribution. Missing entry, invalid/existing label and I/O errors
        propagate. Binding failure retains the new point and old selection for
        use_point recovery; no rollback. The owner manages files/binding.
        Synchronous from a worker or owner callback.
        """
        request = PointCreateRequest(label, "", CommandOrigin("gui"))
        return self.calls.call(lambda: self.control.new_point(request))

    def open_entry(self, name: str) -> EntryInfo:
        """Open an existing entry by single-segment name; do not select a point.

        Owner validates name and clears alias rows until a point is selected.
        I/O/binding failures propagate and preserve the old binding.
        Safe from worker/owner callback.
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

        Invalid rows propagate without changing the old table. Point-local
        disk aliases remain governed by Inspect Apply. Worker/owner safe.
        """
        self.calls.call(lambda: self.control.set_alias_rows(rows))

    def apply_alias_rows(self, rows: tuple[AliasRow, ...]) -> None:
        """Validate and persist ordered authored rows, then publish them in memory.

        Save the selected point's aliases.yaml, then install memory. Without a
        selected point raise FailedPreconditionError(no_context) before I/O.
        Unlike set_alias_rows, this commits point-local aliases. Invalid
        rows and storage errors propagate, preserving the prior table/file.
        The context owner manages validation and atomic file replacement.
        Synchronous from a worker or owner callback.
        """
        self.calls.call(lambda: self.control.apply_alias_rows(rows))

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

    def current(self) -> tuple[StartupTab, ...]:
        """Read detached ordered adapter/input projections without recapture or I/O.

        Worker/owner safe. Inputs are the Form owner's opaque authored encoding,
        not results; later GUI edits do not change these copies. Caller owns them.
        """
        return self.calls.call(self.control.read_inputs)

    def new_tab(self, adapter: str, *, input: TabInput | None = None) -> str:
        """Create adapter's tab, capture ready sources once, and return its ID.

        adapter is a registered catalog name. input optionally transfers opaque
        FormInput.to_json data, preserving drafts/selections on fresh sources.
        Incompatible paths are discarded by Form restore and logged. Unknown
        adapters, malformed input and form/source failures propagate before
        publication. Earlier tabs remain. Worker/owner safe; caller owns input,
        GUI owns the created tab. No file I/O or pending restoration occurs.
        """
        return self.calls.call(lambda: self.control.new_tab(adapter, input=input))


@dataclass(frozen=True)
class StartupContext:
    """Capabilities passed to trusted startup(ctx), without persistent session state.

    soc installs caller-owned boards. devices transfers successfully registered
    drivers to GUI ownership. setup creates or selects entry/point. context edits
    aliases/components.
    tabs creates tabs from ready sources and reads authored
    inputs. calls selects the session owner. register_close records callbacks in
    the runner; report_progress publishes progress and report_summary sets final
    status. These callbacks are owner commands reached through calls.
    cache_root is the absolute user cache directory for scripts to choose their
    own files; it need not exist. Scripts own file schemas, I/O and errors.
    Queries and nonblocking commands support worker/owner callbacks. Blocking
    soc/device connect is worker-only and rejects owner calls before admission.
    """

    soc: StartupSoc
    devices: StartupDevices
    setup: StartupSetup
    context: StartupParameters
    tabs: StartupTabs
    calls: OwnerCalls
    register_close: Callable[[Callable[[], None]], None]
    report_progress: Callable[[str], None]
    report_summary: Callable[[str], None]
    cache_root: Path

    def summary(self, message: str) -> None:
        """Set the final single-line status for successful startup terminal delivery.

        Worker/owner safe while startup runs. Blank or multiline messages raise
        ValueError; calls after terminal raise RuntimeError. Last call wins.
        Does not display a dialog or hide an unhandled script failure. No I/O.
        """
        self.calls.call(lambda: self.report_summary(message))

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
