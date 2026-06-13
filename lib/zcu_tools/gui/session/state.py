"""SessionState — the session-core slice of GUI app state.

Holds what every measurement-session app shares: the active ``ExpContext``, the
multi-device set (``DeviceState`` keyed by name), the remembered startup prefs,
and the optimistic-concurrency ``VersionTable`` (a single shared table — each app
adds its own experiment-surface keys to the same table, decision 6). An app's
own ``State`` subclasses this and adds its experiment slice (measure: tabs; a
sibling app: its own surface), so ``state.exp_context`` / ``state.devices`` /
``state.version`` resolve uniformly across apps.

Import-clean: ``BaseDeviceInfo`` is referenced only under TYPE_CHECKING (its
module pulls in matplotlib.pyplot at import time, which would break the GUI's
"configure the backend before pyplot import" invariant if loaded eagerly).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from zcu_tools.gui.session.types import ExpContext
from zcu_tools.gui.version_table import VersionTable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.device.base import BaseDeviceInfo


class DeviceStatus(Enum):
    """Lifecycle status of a device entry held in State.

    ``MEMORY_ONLY`` is a remembered-but-not-live entry (no driver in
    GlobalDeviceManager). All other values are *live* statuses: the device has
    a driver registered and is either idle (``CONNECTED``) or in a transient
    operation. The cross-object invariant is: a device is in GlobalDeviceManager
    iff its State status is a live status.
    """

    MEMORY_ONLY = "memory_only"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    SETTING_UP = "setting_up"


@dataclass(frozen=True)
class DeviceState:
    """Serializable device state — the SSOT for one device.

    State owns this; DeviceService holds only the live driver (in
    GlobalDeviceManager), the worker threads and the progress model. ``info`` is
    a ``BaseDeviceInfo`` value snapshot (not a live driver) and so lives here.
    ``remember`` is the persistent flag that drives the startup persistence
    projection — it is no longer a transient connect-request attribute.

    There is deliberately no ``progress`` field: setup progress is live
    telemetry owned by ``ProgressService`` (keyed by the operation token) and is
    polled separately via ``operation.progress`` (by operation_id) — never
    spliced into state.
    """

    name: str
    type_name: str
    address: str
    status: DeviceStatus
    remember: bool
    info: BaseDeviceInfo | None = None
    error: str | None = None

    # -- status predicates (the entity answers questions about itself) -----

    def is_memory_only(self) -> bool:
        return self.status is DeviceStatus.MEMORY_ONLY

    def is_connected(self) -> bool:
        """Idle and live (a driver is registered, no operation in flight)."""
        return self.status is DeviceStatus.CONNECTED

    def is_live(self) -> bool:
        """A driver is registered (connected or in a transient operation) — the
        cross-object invariant's 'in GlobalDeviceManager' side."""
        return self.status is not DeviceStatus.MEMORY_ONLY


# Set-cardinality version key for the whole device collection. Bumped only when
# the device *set* gains or loses a member (not on status/info edits of an
# existing member — those move that member's own ``device:<name>`` key). A
# guarded op depending on the whole set (run.start) declares this key so a
# concurrently-added device is detected, which a per-member glob cannot reveal.
DEVICE_SET_VERSION_KEY = "devices:__set__"

DEFAULT_LEFT_PANEL_WIDTH = 500


@dataclass
class StartupPrefs:
    """Remembered startup preferences — the *prefill* values, distinct from the
    active ``ExpContext``.

    These are what the setup dialog prefills and what persistence projects to
    disk; they are NOT the live connection/active-project state. Because the
    instrument never auto-connects on launch, there is no need to distinguish
    "currently connected to" from "remembered" — apply/connect just update these
    prefill values at write-time, and restore writes them back without applying
    a context. Mutable (State holds live mutable objects); a value-only block, so
    not versioned (no guarded op depends on it).
    """

    chip_name: str = ""
    qub_name: str = ""
    res_name: str = ""
    result_dir: str = ""
    database_path: str = ""
    ip: str = "192.168.10.1"
    port: int = 8887
    left_panel_width: int = DEFAULT_LEFT_PANEL_WIDTH


class SessionState:
    """Passive session-core state — the active context, the device set, the
    remembered startup prefs, and the shared version table. App ``State``
    subclasses add their experiment-surface slice + version keys."""

    def __init__(self, ctx: ExpContext) -> None:
        self.exp_context: ExpContext = ctx
        # Remembered startup prefs (prefill values), distinct from exp_context.
        # StartupService writes at apply/connect; PersistenceCaretaker projects.
        self.startup_prefs: StartupPrefs = StartupPrefs()
        # Device state SSOT. DeviceService writes here (on the Qt main thread,
        # at its terminal slots) and holds only the live driver / worker / progress.
        self.devices: dict[str, DeviceState] = {}
        # Optimistic-concurrency version counters. Owned here but bumped by
        # whichever main-thread writer actually mutates a resource (these device
        # mutators for device:<name>; connection/context services for soc/context
        # at their terminal slots; each app's experiment writers for their keys).
        self.version = VersionTable()

    def set_context(self, ctx: ExpContext) -> None:
        """Replace the whole ExpContext. Pure field swap — does NOT bump the
        ``context`` resource version, because the same setter is used to swap
        non-md/ml fields (soc/soccfg via connect, predictor via load/clear).

        The ``context`` version represents md/ml *content*: only callers that
        actually change md/ml (setup_project / use_context / new_context) bump
        it explicitly. soc has its own ``soc`` version key; predictor is not a
        guarded resource. This keeps soc-connect / predictor-load from spuriously
        marking md/ml-dependent ops (run / editor.commit / writeback) stale.

        The full set of "writes md/ml → bump context" paths is enumerated at the
        canonical anchor on ``ContextService.set_md_attr``.
        """
        self.exp_context = ctx

    # ------------------------------------------------------------------
    # Device state (DeviceService writes these on the Qt main thread).
    # Every *semantic write* bumps device:<name>; cache refresh does not
    # (see refresh_device_info_cache).
    # ------------------------------------------------------------------

    def put_device(self, dev: DeviceState) -> None:
        """Insert or replace a device entry (create / status transition)."""
        logger.debug("put_device: name=%r status=%s", dev.name, dev.status.value)
        is_new = dev.name not in self.devices
        self.devices[dev.name] = dev
        self.version.bump(f"device:{dev.name}")
        # Set-membership guard: a per-device version cannot reveal a *new* member
        # to an opt-in concurrency check (the agent never declared a key for a
        # device that did not exist when it read versions). The set-cardinality
        # key advances whenever the device set grows or shrinks, so an op that
        # depends on the whole set (run.start) detects a concurrently-added
        # device. Status transitions reuse an existing entry → set unchanged →
        # no bump here (the per-device key already moves).
        if is_new:
            self.version.bump(DEVICE_SET_VERSION_KEY)

    def set_device_status(
        self, name: str, status: DeviceStatus, *, error: str | None = None
    ) -> None:
        logger.debug("set_device_status: name=%r status=%s", name, status.value)
        self.devices[name] = replace(self.devices[name], status=status, error=error)
        self.version.bump(f"device:{name}")

    def set_device_info(self, name: str, info: BaseDeviceInfo | None) -> None:
        """Semantic info update (after set-value / setup); bumps version."""
        logger.debug("set_device_info: name=%r info=%s", name, info is not None)
        self.devices[name] = replace(self.devices[name], info=info)
        self.version.bump(f"device:{name}")

    def set_device_remember(self, name: str, remember: bool) -> None:
        logger.debug("set_device_remember: name=%r remember=%s", name, remember)
        self.devices[name] = replace(self.devices[name], remember=remember)
        self.version.bump(f"device:{name}")

    def refresh_device_info_cache(self, name: str, info: BaseDeviceInfo) -> bool:
        """Refresh the cached driver info on a *read*; return whether it changed.

        A live read (``get_device_info``) caches the freshest driver info back
        into State. The principle is "bump == state actually changed", not "bump
        == a client wrote":

        - **info unchanged** → pure cache sync; do NOT bump (a redundant refresh
          must not spuriously invalidate other clients' ``expected_versions``).
        - **info changed** → the device's real value moved underneath us (e.g.
          external hardware change). That *is* a genuine state change, so bump
          ``device:<name>`` and return True so the caller can emit DEVICE_CHANGED.
          Compared by full ``BaseDeviceInfo`` value equality (pydantic ``==``).
        """
        current = self.devices[name]
        changed = current.info != info
        self.devices[name] = replace(current, info=info, error=None)
        if changed:
            self.version.bump(f"device:{name}")
        return changed

    def remove_device(self, name: str) -> None:
        logger.debug("remove_device: name=%r", name)
        del self.devices[name]
        # A stale dependency on a removed device now reads as version 0 (gone).
        self.version.drop_prefix(f"device:{name}")
        # The device set shrank — advance the set-cardinality key (symmetric to
        # the grow case in put_device) so a whole-set dependant detects it.
        self.version.bump(DEVICE_SET_VERSION_KEY)

    def get_device(self, name: str) -> DeviceState | None:
        return self.devices.get(name)

    def has_device(self, name: str) -> bool:
        return name in self.devices

    def list_devices(self) -> tuple[DeviceState, ...]:
        return tuple(self.devices[name] for name in sorted(self.devices))
