from __future__ import annotations

import threading
import warnings
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from zcu_tools.cfg_model import validate_device_name

from .base import BaseDevice, DeviceActivity

if TYPE_CHECKING:
    from . import DeviceInfo


class DeviceCloseFailure(RuntimeError):
    """A registry-owned ``BaseDevice.close()`` raised an ordinary ``Exception``.

    ``names`` contains the requested registry name whose driver failed to
    close, as a one-element tuple. The entry stays registered so the caller
    can retry; the in-flight claim is already released.
    """

    def __init__(self, names: tuple[str, ...], cause: Exception) -> None:
        super().__init__(f"Device close failed for {', '.join(names)}: {cause}")
        self.names = names
        self.cause = cause


class DeviceCloseInProgressError(RuntimeError):
    """A manager close API already holds the in-flight claim for this identity.

    Manager close and registration APIs fail fast instead of waiting for an
    in-flight close. ``names`` contains the rejected request's registry name
    as a one-element tuple.
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__(f"Device close already in progress for {', '.join(names)}")
        self.names = names


class DeviceManager:
    """Own a named device registry and coordinate its drivers' lifetimes.

    Share one manager between callers that share drivers. Separate managers
    isolate registrations and close claims; they must not own the same driver.
    Disconnect never changes device outputs or closes the resource factory.
    """

    def __init__(self) -> None:
        self._devices: dict[str, BaseDevice[Any]] = {}
        self._lock = threading.RLock()
        # Keep claimed identities alive until their close completes.
        self._close_claims: dict[int, object] = {}

    def register_device(self, name: str, device: BaseDevice[Any]) -> None:
        """Register a driver under its only name in this manager.

        Raise ``TypeError`` for a non-``BaseDevice`` input and ``ValueError``
        if name is empty, whitespace-only, contains an ASCII dot, or this driver
        is already registered under a different name. Invalid-name errors include
        the rejected name. Raise ``DeviceCloseInProgressError`` if a manager close
        API holds a claim
        on this driver, even after its original name was dropped or replaced.
        Otherwise, reusing a name warns and overwrites its entry, including
        for the same driver. Rejections leave the registry unchanged.
        Valid names are used unchanged, including surrounding whitespace.
        Registration performs no device I/O. The manager coordinates the driver's
        lifetime; retrieval only borrows the registered object.
        """
        # tolerated: Reject non-BaseDevice runtime inputs before changing the registry.
        if not isinstance(device, BaseDevice):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"register_device expected BaseDevice for {name!r}, "
                f"got {type(device).__name__}"
            )

        validate_device_name(name)
        with self._lock:
            if id(device) in self._close_claims:
                raise DeviceCloseInProgressError((name,))
            for registered_name, registered in self._devices.items():
                if registered is device and registered_name != name:
                    raise ValueError(
                        f"Device already registered as {registered_name!r}; "
                        f"cannot register as {name!r}"
                    )
            if name in self._devices:
                warnings.warn(
                    f"Device {name} already registered, overwriting", stacklevel=1
                )
            self._devices[name] = device

    def drop_device(self, name: str, *, ignore_error: bool = False) -> None:
        with self._lock:
            if name not in self._devices:
                if ignore_error:
                    return
                raise ValueError(f"Device {name} not found")
            del self._devices[name]

    def get_device(self, name: str) -> BaseDevice[Any]:
        """Borrow the same registered driver for the exact string name.

        Return the original object without device I/O or transfer of lifecycle
        ownership. Raise ValueError if name is not registered.
        """
        with self._lock:
            if name not in self._devices:
                raise ValueError(f"Device {name} not found")
            return self._devices[name]

    def get_all_devices(self) -> dict[str, BaseDevice[Any]]:
        """Return an independent name dictionary sharing the registered drivers.

        Keys retain the exact registered strings. Changing this dictionary
        does not change the registry. Driver references are
        borrowed, not copied; this performs no device I/O and does not transfer
        lifecycle ownership.
        """
        with self._lock:
            return dict(self._devices)

    def activity_snapshot(self) -> dict[str, DeviceActivity]:
        """Return one detached activity value per currently registered name.

        Any thread may call this method. Names/drivers are copied under the
        registry lock; activity is then sampled without hardware I/O or waiting
        for operation/I/O locks. Later registry edits do not change the returned
        mapping. Driver fields and different drivers are not sampled atomically.
        See DeviceActivity for field meanings, initial values and monotonic
        seconds. Empty registries return an empty dict; busy drivers do not fail
        this observation. The returned dict is caller-owned, its values are
        immutable, and neither can modify driver or registry state.
        """
        return {
            name: device.activity for name, device in self.get_all_devices().items()
        }

    def setup_devices(
        self,
        dev_cfg: Mapping[str, DeviceInfo],
        *,
        progress: bool = True,
        cancel_signal: threading.Event | None = None,
    ) -> None:
        # Validate all names and snapshot references under the registry lock so
        # that the check-then-act is atomic with respect to concurrent
        # register/drop calls.  Fast-fail: any unknown name aborts the whole
        # batch before any setup begins.
        with self._lock:
            for name in dev_cfg:
                if name not in self._devices:
                    raise ValueError(f"Device {name} not found")
            # Snapshot instance references; registry mutations after this point
            # do not affect which instances we are about to configure.
            snapshot: list[tuple[BaseDevice[Any], DeviceInfo]] = [
                (self._devices[name], cfg) for name, cfg in dev_cfg.items()
            ]

        # Per-instance op_lock serializes each setup() call. Busy devices raise
        # DeviceBusyError immediately (fail-fast); we do not swallow that error.
        for device, cfg in snapshot:
            if cancel_signal is not None and cancel_signal.is_set():
                return
            device.setup(
                cfg,
                progress=progress,
                stop_event=cancel_signal,
            )

    def get_info(self, name: str) -> DeviceInfo:
        # Resolve the instance under the registry lock; call get_info() outside
        # it so a long-running setup() on another device cannot block this read.
        device = self.get_device(name)
        return device.get_info()

    def get_all_info(self) -> dict[str, DeviceInfo]:
        # Snapshot the registry under the lock, then query each device outside
        # it so concurrent setup() calls on individual devices do not block
        # the whole registry for the duration of their I/O.
        snapshot = self.get_all_devices()
        return {name: device.get_info() for name, device in snapshot.items()}

    # ------------------------------------------------------------------
    # Registry-owned disconnect
    # ------------------------------------------------------------------

    def close_device(self, name: str, *, ignore_missing: bool = False) -> None:
        """Close one registered device and remove its entry on success.

        The call claims the device identity (``id(device)``) under the registry
        lock; the actual ``device.close()`` runs outside the lock. A second
        manager close API targeting the same in-flight identity raises
        ``DeviceCloseInProgressError`` immediately instead of waiting or
        double-closing. On success the requested entry is removed only if it
        still refers to this driver, so a same-name replacement survives.
        On ordinary failure the entry stays registered for retry.
        """
        with self._lock:
            if name not in self._devices:
                if ignore_missing:
                    return
                raise ValueError(f"Device {name} not found")
            device = self._devices[name]
            identity = id(device)
            if identity in self._close_claims:
                raise DeviceCloseInProgressError((name,))
            self._close_claims[identity] = device

        try:
            device.close()
        except Exception as exc:
            self._release_close_claim(identity)
            raise DeviceCloseFailure((name,), exc) from exc
        except BaseException:
            self._release_close_claim(identity)
            raise
        else:
            # Remove the entry and release its claim atomically so a follower
            # cannot reclaim a successfully closed, still-registered driver.
            self._finish_close(
                closed_entries=((name, identity),),
                claimed_identities=(identity,),
            )

    def close_all_devices(self) -> None:
        """Close every registered device once, aggregating named errors.

        Snapshot registrations and claim all available drivers in one atomic
        registry-lock operation, then run each ``device.close()`` outside the
        lock. Ordinary failures are collected as ``DeviceCloseFailure`` and
        drivers already claimed by a concurrent manager close API fail fast
        as ``DeviceCloseInProgressError``. Both are aggregated in one built-in
        ``ExceptionGroup`` while the rest of the batch still runs. Failed
        entries stay registered for retry. Successful entries are removed
        only if they still refer to the snapshotted driver. A ``BaseException``
        propagates unwrapped, but only after successful entries are cleaned
        up and every owned claim is released. An empty registry is a no-op.
        """
        # Snapshot and claim together: a follower sees either an in-flight
        # claim or an entry already removed by the successful close.
        with self._lock:
            snapshot = dict(self._devices)
            if not snapshot:
                return

            claimed: dict[str, BaseDevice[Any]] = {}
            in_progress: list[DeviceCloseInProgressError] = []
            for name, device in snapshot.items():
                identity = id(device)
                if identity in self._close_claims:
                    # Another manager close API owns this identity: fail fast
                    # with a named error and let the batch continue.
                    in_progress.append(DeviceCloseInProgressError((name,)))
                    continue
                self._close_claims[identity] = device
                claimed[name] = device

        failures: list[DeviceCloseFailure] = []
        succeeded: list[tuple[str, int]] = []
        try:
            for name, device in claimed.items():
                try:
                    device.close()
                # tolerated: Drivers may raise any ordinary error; aggregate it and close the rest.
                except Exception as exc:  # noqa: BLE001
                    failures.append(DeviceCloseFailure((name,), exc))
                else:
                    succeeded.append((name, id(device)))
        finally:
            # A BaseException propagating from a later device must not leave
            # earlier successful closes registered, nor any owned claim held:
            # clean both up atomically before the exception escapes.
            self._finish_close(
                closed_entries=succeeded,
                claimed_identities=(id(device) for device in claimed.values()),
            )

        errors: list[Exception] = [*failures, *in_progress]
        if errors:
            raise ExceptionGroup(f"failed to close {len(errors)} device(s)", errors)

    def _release_close_claim(self, identity: int) -> None:
        with self._lock:
            self._close_claims.pop(identity, None)

    def _finish_close(
        self,
        *,
        closed_entries: Iterable[tuple[str, int]],
        claimed_identities: Iterable[int],
    ) -> None:
        """Atomically remove successful entries and release owned claims.

        ``closed_entries`` pairs each successfully closed name with its driver
        identity. Remove only a still-matching entry to preserve replacements.
        ``claimed_identities`` identifies the claims owned by this close call.
        Keeping cleanup and claim release under one lock prevents a follower
        from reclaiming a successfully closed, still-registered driver.
        """
        with self._lock:
            for name, identity in closed_entries:
                if name in self._devices and id(self._devices[name]) == identity:
                    del self._devices[name]
            for identity in claimed_identities:
                self._close_claims.pop(identity, None)
