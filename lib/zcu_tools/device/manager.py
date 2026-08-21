from __future__ import annotations

import threading
import warnings
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, ClassVar

from .base import BaseDevice
from .cancel_scope import current_device_setup_cancel_signal

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


class GlobalDeviceManager:
    _devices: ClassVar[dict[str, BaseDevice]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()
    # In-flight close claims keyed by id(device).  The value holds a strong
    # reference so the identity cannot be recycled while a close is pending.
    _close_claims: ClassVar[dict[int, BaseDevice]] = {}

    @classmethod
    def register_device(cls, name: str, device: BaseDevice) -> None:
        if not isinstance(device, BaseDevice):
            raise TypeError(
                f"register_device expected BaseDevice for {name!r}, "
                f"got {type(device).__name__}"
            )

        with cls._lock:
            if name in cls._devices:
                warnings.warn(f"Device {name} already registered, overwriting")
            cls._devices[name] = device

    @classmethod
    def drop_device(cls, name: str, ignore_error: bool = False) -> None:
        with cls._lock:
            if name not in cls._devices:
                if ignore_error:
                    return
                raise ValueError(f"Device {name} not found")
            del cls._devices[name]

    @classmethod
    def get_device(cls, name: str) -> BaseDevice:
        with cls._lock:
            if name not in cls._devices:
                raise ValueError(f"Device {name} not found")
            return cls._devices[name]

    @classmethod
    def get_all_devices(cls) -> dict[str, BaseDevice]:
        with cls._lock:
            return dict(cls._devices)

    @classmethod
    def setup_devices(
        cls,
        dev_cfg: Mapping[str, DeviceInfo],
        *,
        progress: bool = True,
        cancel_signal: threading.Event | None = None,
    ) -> None:
        # Validate all names and snapshot references under the registry lock so
        # that the check-then-act is atomic with respect to concurrent
        # register/drop calls.  Fast-fail: any unknown name aborts the whole
        # batch before any setup begins.
        with cls._lock:
            for name in dev_cfg:
                if name not in cls._devices:
                    raise ValueError(f"Device {name} not found")
            # Snapshot instance references; registry mutations after this point
            # do not affect which instances we are about to configure.
            snapshot: list[tuple[BaseDevice, DeviceInfo]] = [
                (cls._devices[name], cfg) for name, cfg in dev_cfg.items()
            ]

        resolved_cancel_signal = (
            cancel_signal
            if cancel_signal is not None
            else current_device_setup_cancel_signal()
        )

        # Per-instance op_lock serializes each setup() call. Busy devices raise
        # DeviceBusyError immediately (fail-fast); we do not swallow that error.
        for device, cfg in snapshot:
            if resolved_cancel_signal is not None and resolved_cancel_signal.is_set():
                return
            device.setup(
                cfg,
                progress=progress,
                stop_event=resolved_cancel_signal,
            )

    @classmethod
    def get_info(cls, name: str) -> DeviceInfo:
        # Resolve the instance under the registry lock; call get_info() outside
        # it so a long-running setup() on another device cannot block this read.
        device = cls.get_device(name)
        return device.get_info()  # type: ignore[return-value]

    @classmethod
    def get_all_info(cls) -> dict[str, DeviceInfo]:
        # Snapshot the registry under the lock, then query each device outside
        # it so concurrent setup() calls on individual devices do not block
        # the whole registry for the duration of their I/O.
        snapshot = cls.get_all_devices()
        return {name: device.get_info() for name, device in snapshot.items()}  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Registry-owned disconnect
    # ------------------------------------------------------------------

    @classmethod
    def close_device(cls, name: str, *, ignore_missing: bool = False) -> None:
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
        with cls._lock:
            if name not in cls._devices:
                if ignore_missing:
                    return
                raise ValueError(f"Device {name} not found")
            device = cls._devices[name]
            identity = id(device)
            if identity in cls._close_claims:
                raise DeviceCloseInProgressError((name,))
            cls._close_claims[identity] = device

        try:
            device.close()
        except Exception as exc:
            cls._release_close_claim(identity)
            raise DeviceCloseFailure((name,), exc) from exc
        except BaseException:
            cls._release_close_claim(identity)
            raise
        else:
            # Release the claim and drop the closed identity's aliases in one
            # atomic lock acquisition: a follower can never observe a released
            # claim whose identity is still registered and double-close it.
            cls._finish_close(
                closed_identities=(identity,),
                claimed_identities=(identity,),
            )

    @classmethod
    def close_all_devices(cls) -> None:
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
        with cls._lock:
            snapshot = dict(cls._devices)
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
                if identity in cls._close_claims:
                    # Another manager close API owns this identity: fail fast
                    # with a named error and let the batch continue.
                    in_progress.append(DeviceCloseInProgressError(tuple(aliases)))
                    continue
                device = devices_by_identity[identity]
                cls._close_claims[identity] = device
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
            cls._finish_close(
                closed_identities=succeeded,
                claimed_identities=(identity for identity, _ in claimed),
            )

        errors: list[Exception] = [*failures, *in_progress]
        if errors:
            raise ExceptionGroup(f"failed to close {len(errors)} device(s)", errors)

    @classmethod
    def _release_close_claim(cls, identity: int) -> None:
        with cls._lock:
            cls._close_claims.pop(identity, None)

    @classmethod
    def _finish_close(
        cls,
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
        with cls._lock:
            if closed:
                for alias in [
                    a for a, dev in cls._devices.items() if id(dev) in closed
                ]:
                    del cls._devices[alias]
            for identity in claimed_identities:
                cls._close_claims.pop(identity, None)
