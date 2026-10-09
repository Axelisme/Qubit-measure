from __future__ import annotations

import threading
import warnings
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from .base import BaseDevice

if TYPE_CHECKING:
    from . import DeviceInfo


class DeviceCloseFailure(RuntimeError):
    """A registry-owned ``BaseDevice.close()`` raised an ordinary ``Exception``.

    ``names`` identifies the registry entries whose identity failed to close
    (all aliases in a batch close; the requested name in a single close).  The
    entries stay registered so the caller can retry; the in-flight claim is
    already released.
    """

    def __init__(self, names: tuple[str, ...], cause: Exception) -> None:
        super().__init__(f"Device close failed for {', '.join(names)}: {cause}")
        self.names = names
        self.cause = cause


class DeviceCloseInProgressError(RuntimeError):
    """A manager close API already holds the in-flight claim for this identity.

    Manager close APIs fail fast: the second caller raises immediately instead
    of waiting for the in-flight close or closing the same device again.
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
        self._devices: dict[str, BaseDevice] = {}
        self._lock = threading.RLock()
        # Keep claimed identities alive until their close completes.
        self._close_claims: dict[int, BaseDevice] = {}

    def register_device(self, name: str, device: BaseDevice) -> None:
        if not isinstance(device, BaseDevice):
            raise TypeError(
                f"register_device expected BaseDevice for {name!r}, "
                f"got {type(device).__name__}"
            )

        with self._lock:
            if name in self._devices:
                warnings.warn(
                    f"Device {name} already registered, overwriting", stacklevel=1
                )
            self._devices[name] = device

    def drop_device(self, name: str, ignore_error: bool = False) -> None:
        with self._lock:
            if name not in self._devices:
                if ignore_error:
                    return
                raise ValueError(f"Device {name} not found")
            del self._devices[name]

    def get_device(self, name: str) -> BaseDevice:
        with self._lock:
            if name not in self._devices:
                raise ValueError(f"Device {name} not found")
            return self._devices[name]

    def get_all_devices(self) -> dict[str, BaseDevice]:
        with self._lock:
            return dict(self._devices)

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
            snapshot: list[tuple[BaseDevice, DeviceInfo]] = [
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
        return device.get_info()  # type: ignore[return-value]

    def get_all_info(self) -> dict[str, DeviceInfo]:
        # Snapshot the registry under the lock, then query each device outside
        # it so concurrent setup() calls on individual devices do not block
        # the whole registry for the duration of their I/O.
        snapshot = self.get_all_devices()
        return {name: device.get_info() for name, device in snapshot.items()}  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Registry-owned disconnect
    # ------------------------------------------------------------------

    def close_device(self, name: str, *, ignore_missing: bool = False) -> None:
        """Close one registered device and drop its stale aliases on success.

        The call claims the device identity (``id(device)``) under the registry
        lock; the actual ``device.close()`` runs outside the lock.  A second
        manager close API targeting the same in-flight identity raises
        ``DeviceCloseInProgressError`` immediately instead of waiting or
        double-closing.  On success every registry alias still pointing at the
        closed identity is removed (a same-name replacement with a different
        identity survives); on ordinary failure the entries are kept so the
        caller can retry.
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
            # Release the claim and drop the closed identity's aliases in one
            # atomic lock acquisition: a follower can never observe a released
            # claim whose identity is still registered and double-close it.
            self._finish_close(
                closed_identities=(identity,),
                claimed_identities=(identity,),
            )

    def close_all_devices(self) -> None:
        """Close every registered device once, aggregating named errors.

        Snapshot, identity-dedupe and claim all claimable identities in one
        atomic registry-lock operation, then run each ``device.close()``
        outside the lock.  Ordinary failures are collected as
        ``DeviceCloseFailure`` and identities already claimed by a concurrent
        manager close API fail fast as ``DeviceCloseInProgressError``; both
        are aggregated in one built-in ``ExceptionGroup`` while the rest of
        the batch still runs.  Entries of failed identities stay registered
        for retry; aliases of successfully closed identities -- including
        same-identity aliases added while the close was in flight -- are
        removed.  A ``BaseException`` propagates unwrapped, but only after
        already-closed identities are cleaned up and every owned claim is
        released.  An empty registry is a no-op.
        """
        # Snapshot, identity-dedupe and claim all in one atomic lock
        # acquisition: a concurrent batch that finished before our lock grab
        # left an already-cleaned registry (so the snapshot is empty or lacks
        # the identity), while one still in flight still holds the claim — a
        # follower can never double-close an identity it snapshotted.
        with self._lock:
            snapshot = dict(self._devices)
            if not snapshot:
                return

            names_by_identity: dict[int, list[str]] = {}
            devices_by_identity: dict[int, BaseDevice] = {}
            for alias, dev in snapshot.items():
                names_by_identity.setdefault(id(dev), []).append(alias)
                devices_by_identity.setdefault(id(dev), dev)

            claimed: list[tuple[int, BaseDevice]] = []
            in_progress: list[DeviceCloseInProgressError] = []
            for identity, aliases in names_by_identity.items():
                if identity in self._close_claims:
                    # Another manager close API owns this identity: fail fast
                    # with a named error and let the batch continue.
                    in_progress.append(DeviceCloseInProgressError(tuple(aliases)))
                    continue
                device = devices_by_identity[identity]
                self._close_claims[identity] = device
                claimed.append((identity, device))

        failures: list[DeviceCloseFailure] = []
        succeeded: list[int] = []
        try:
            for identity, device in claimed:
                try:
                    device.close()
                except Exception as exc:
                    failures.append(
                        DeviceCloseFailure(tuple(names_by_identity[identity]), exc)
                    )
                else:
                    succeeded.append(identity)
        finally:
            # A BaseException propagating from a later device must not leave
            # earlier successful closes registered, nor any owned claim held:
            # clean both up atomically before the exception escapes.
            self._finish_close(
                closed_identities=succeeded,
                claimed_identities=(identity for identity, _ in claimed),
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
        closed_identities: Iterable[int],
        claimed_identities: Iterable[int],
    ) -> None:
        """Atomically drop closed-identity aliases and release owned claims.

        One registry-lock acquisition covers both mutations so a follower
        observes either the in-flight claim (``DeviceCloseInProgressError``)
        or a fully cleaned registry -- never a released claim whose identity
        is still registered.  ``closed_identities`` are identities whose
        ``close()`` returned successfully; every remaining alias pointing at
        them is removed.  ``claimed_identities`` are the claims this call
        owns and must release.
        """
        closed = set(closed_identities)
        with self._lock:
            if closed:
                for alias in [
                    a for a, dev in self._devices.items() if id(dev) in closed
                ]:
                    del self._devices[alias]
            for identity in claimed_identities:
                self._close_claims.pop(identity, None)
