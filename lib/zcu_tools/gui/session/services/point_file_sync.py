"""Installed-versus-disk revision admission and selected-point observations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from ruamel.yaml.error import YAMLError

from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.owner import OwnerThreadGuard
from zcu_tools.gui.session.context_types import PointFileRevision, PointFileSyncStatus
from zcu_tools.gui.session.events import PointFileChangedPayload


class PointFileSync:
    """Coordinate revisions and observations, without interpreting document formats.

    ContextService supplies installed versions and one strict installation
    command. It retains resource validation and content publication authority.
    UI draft checks, RPC reload and watcher scans share this single revision
    policy. Calls run on the constructing owner thread.
    """

    def __init__(
        self,
        bus: BaseEventBus,
        *,
        installed: Callable[[], tuple[PointFileRevision, ...]],
        install: Callable[[PointFileRevision], PointFileRevision],
    ) -> None:
        """Bind installed queries and strict install(file) to the context owner.

        installed performs no I/O. install reads/validates a candidate without
        repairing disk, retains memory on rejection, and publishes content facts
        only after installation. This coordinator publishes observation facts.
        """
        self._owner = OwnerThreadGuard()
        self._bus = bus
        self._installed = installed
        self._install = install
        self._statuses: dict[Path, PointFileSyncStatus] = {}

    def clear(self) -> None:
        """Forget observations on entry/point switch; never touch files."""
        self._owner.assert_owner()
        self._statuses.clear()

    def statuses(self) -> tuple[PointFileSyncStatus, ...]:
        """Project cached observations with current installed versions, without I/O.

        disk_revision retains the last observation, even after a later owner
        write; file always reflects the current installed owner version.
        """
        self._owner.assert_owner()
        installed = {file.path: file for file in self._installed()}
        return tuple(
            replace(status, file=installed[path])
            for path, status in self._statuses.items()
            if path in installed
        )

    def _point_file(self, path: Path) -> PointFileRevision:
        if path.is_absolute():
            for file in self._installed():
                if file.path == path:
                    return file
        raise InvalidInputError(
            f"{path}: not an absolute owner path of the selected point",
            reason_code="file_path_invalid",
        )

    @staticmethod
    def _validate_base(file: PointFileRevision, revision: str | None) -> None:
        if revision is None and file.document != "aliases":
            raise InvalidInputError(
                f"{file.path}: a revision is required for {file.document}"
            )

    @staticmethod
    def _disk_revision(path: Path) -> str | None:
        try:
            return sha256(path.read_bytes()).hexdigest()
        except FileNotFoundError:
            return None

    def check_revision(self, expected: PointFileRevision) -> None:
        """Compare a draft's absolute owner path/version with disk, without effects.

        None expects an absent aliases file only. Invalid paths/bases raise
        InvalidInputError. Mismatch raises FailedPreconditionError with reason
        revision_mismatch. Other I/O failures propagate; no candidate is installed.
        """
        self._owner.assert_owner()
        file = self._point_file(expected.path)
        if expected.document != file.document:
            raise InvalidInputError(f"{expected.path}: document owner mismatch")
        self._validate_base(file, expected.revision)
        if self._disk_revision(file.path) != expected.revision:
            raise FailedPreconditionError(
                f"{file.path}: disk revision mismatch", reason_code="revision_mismatch"
            )

    def _record(self, status: PointFileSyncStatus) -> None:
        if self._statuses.get(status.file.path) != status:
            self._statuses[status.file.path] = status
            self._bus.emit(PointFileChangedPayload(status=status))

    def reload(self, path: Path, expected_revision: str | None) -> PointFileRevision:
        """Check the installed base, strictly install the file, then publish status.

        None expects an absent aliases base only. Invalid paths/bases/candidates
        raise InvalidInputError with path/reason. A stale installed base raises
        FailedPreconditionError(revision_mismatch). Candidate failure preserves
        memory/disk and publishes only an error observation. Caller owns holder
        admission; this method never bypasses the injected installation command.
        """
        self._owner.assert_owner()
        file = self._point_file(path)
        self._validate_base(file, expected_revision)
        if file.revision != expected_revision:
            raise FailedPreconditionError(
                f"{path}: installed revision mismatch", reason_code="revision_mismatch"
            )
        try:
            installed = self._install(file)
        except (InvalidInputError, ValueError, OSError, YAMLError) as exc:
            error = f"{path}: {exc}"
            try:
                disk_revision = self._disk_revision(path)
            except OSError:
                disk_revision = None
            self._record(PointFileSyncStatus(file, disk_revision, "error", error))
            raise InvalidInputError(error, reason_code="file_validation") from exc
        self._record(
            PointFileSyncStatus(installed, installed.revision, "reloaded", None)
        )
        return installed

    def synchronize(self, *, admit_reload: bool) -> None:
        """Observe all selected-point files; admit external installation only if True.

        False publishes pending without changing installed resources. Equal hashes
        ignore own writes. Duplicate observations are silent. Validation and I/O
        errors publish per-file diagnostics and do not stop the remaining scan.
        """
        self._owner.assert_owner()
        for file in self._installed():
            try:
                disk_revision = self._disk_revision(file.path)
            except OSError as exc:
                self._record(
                    PointFileSyncStatus(file, None, "error", f"{file.path}: {exc}")
                )
                continue
            if disk_revision == file.revision:
                previous = self._statuses.get(file.path)
                if previous is not None and previous.state != "reloaded":
                    self._record(
                        PointFileSyncStatus(file, disk_revision, "reloaded", None)
                    )
                continue
            if not admit_reload:
                self._record(PointFileSyncStatus(file, disk_revision, "pending", None))
                continue
            try:
                self.reload(file.path, file.revision)
            except InvalidInputError:
                # reload has published the candidate's diagnostic.
                continue
