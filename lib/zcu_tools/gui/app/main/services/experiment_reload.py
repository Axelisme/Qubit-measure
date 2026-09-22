"""Owner-thread experiment replacement with RAM-only tab recovery."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import get_ident
from typing import Protocol

from zcu_tools.gui.app.main.catalog import (
    CatalogReloadError,
    ExperimentAccess,
    ExperimentCatalogLoader,
    PreparedCatalogReload,
)
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.expected_error import FailedPreconditionError

from .persistence_types import PersistedSession, PersistedTab
from .ports import RestoreIssue, RestoreReport


class ReloadWorkspacePort(Protocol):
    def capture_session(self) -> PersistedSession: ...
    def apply_session(self, session: PersistedSession) -> RestoreReport: ...
    def close_tab(self, tab_id: str) -> None: ...


@dataclass(frozen=True, eq=False)
class ReloadPreview:
    """Single-use confirmation identity; no live adapters or results escape."""

    tabs: tuple[tuple[str, str], ...]
    skipped_count: int


@dataclass(frozen=True)
class ReloadReport:
    restored_tabs: int = 0
    rejected_tabs: tuple[RestoreIssue, ...] = ()
    catalog_error: str | None = None
    restart_required: bool = False


class ExperimentReloadService:
    def __init__(
        self,
        *,
        state: State,
        workspace: ReloadWorkspacePort,
        registry: Registry,
        loader: ExperimentCatalogLoader | None,
        active_operations: Callable[[], int],
        access: ExperimentAccess,
    ) -> None:
        self._state = state
        self._workspace = workspace
        self._registry = registry
        self._loader = loader
        self._active_operations = active_operations
        self._access = access
        self._owner = get_ident()
        self._preview: ReloadPreview | None = None
        self._identity: object = None
        self._recovery: PersistedSession | None = None
        self._skipped: tuple[PersistedTab, ...] = ()
        self._restart_required = False

    @property
    def can_retry_reload(self) -> bool:
        return self._recovery is not None and not self._restart_required

    @property
    def skipped_count(self) -> int:
        return len(self._skipped)

    def _require_idle(self) -> ExperimentCatalogLoader:
        if get_ident() != self._owner:
            raise RuntimeError("Experiment reload must run on the owner thread")
        if self._loader is None:
            raise FailedPreconditionError("No experiment loader is configured")
        if self._access.switching or self._access.shutting_down:
            raise FailedPreconditionError(
                "Reload is unavailable during another switch or shutdown"
            )
        if self._restart_required:
            raise FailedPreconditionError(
                "Restart the app after the previous reload failure"
            )
        # SaveService is not handle-backed: checking only live_count misses it.
        if self._active_operations() or any(
            self._state.is_tab_busy(tab_id) for tab_id in self._state.list_tab_ids()
        ):
            raise FailedPreconditionError(
                "Finish all operations before reloading experiments"
            )
        return self._loader

    def _tab_identity(self) -> object:
        return (
            tuple(self._state.list_tab_ids()),
            self._state.active_tab_id,
            self._state.version.snapshot(),
            tuple(
                id(self._state.get_tab(key).run.result)
                for key in self._state.list_tab_ids()
            ),
        )

    def prepare_reload(self) -> ReloadPreview:
        loader = self._require_idle()
        self._access.require_available()
        loader.prepare()
        self._preview = ReloadPreview(
            tuple(
                (key, self._state.get_tab(key).adapter_name)
                for key in self._state.list_tab_ids()
            ),
            len(self._skipped),
        )
        self._identity = self._tab_identity()
        return self._preview

    def reload_confirmed(self, preview: ReloadPreview) -> ReloadReport:
        loader = self._require_idle()
        self._access.require_available()
        if preview is not self._preview or self._identity != self._tab_identity():
            self._preview = None
            raise FailedPreconditionError(
                "Tabs changed while confirming; confirm reload again"
            )
        self._preview = None
        plan = loader.prepare()
        snapshot = self._workspace.capture_session().model_copy(deep=True)
        self._recovery = snapshot
        self._skipped = ()
        with self._switch():
            for key in tuple(self._state.list_tab_ids()):
                self._workspace.close_tab(key)
            self._registry.clear()
            self._access.available = False
            return self._load_and_restore(loader, plan)

    def retry_failed_reload(self) -> ReloadReport:
        loader = self._require_idle()
        if (
            not self.can_retry_reload
            or self._access.available
            or self._state.list_tab_ids()
        ):
            raise FailedPreconditionError("There is no failed catalog reload to retry")
        plan = loader.prepare()
        with self._switch():
            return self._load_and_restore(loader, plan)

    @contextmanager
    def _switch(self) -> Iterator[None]:
        self._access.switching = True
        try:
            yield
        except Exception as exc:
            self._access.available = False
            self._restart_required = True
            raise CatalogReloadError(
                f"Reload could not finish safely: {exc}. Restart the app; "
                "the RAM recovery snapshot will be lost.",
                restart_required=True,
            ) from exc
        finally:
            self._access.switching = False

    def _load_and_restore(
        self, loader: ExperimentCatalogLoader, plan: PreparedCatalogReload
    ) -> ReloadReport:
        try:
            candidate = loader.load(plan)
        except CatalogReloadError as exc:
            self._restart_required = exc.restart_required
            return ReloadReport(
                catalog_error=str(exc), restart_required=exc.restart_required
            )
        self._registry.replace_from(candidate)
        snapshot = self._recovery
        if snapshot is None:
            raise RuntimeError("Reload lost its RAM snapshot")
        report = self._restore(snapshot)
        self._recovery = None
        self._access.available = True
        return report

    def _restore(self, snapshot: PersistedSession) -> ReloadReport:
        restored = 0
        issues: list[RestoreIssue] = []
        skipped: list[PersistedTab] = []
        active: str | None = None
        for index, entry in enumerate(snapshot.tabs):
            before = set(self._state.list_tab_ids())
            # The existing workspace seam owns raw-cfg decoding and TabAdded.
            report = self._workspace.apply_session(
                PersistedSession(tabs=(entry.model_copy(deep=True),))
            )
            issues.extend(report.rejected_tabs)
            if report.rejected_tabs:
                skipped.append(entry)
            else:
                restored += report.restored_tabs
                added = [key for key in self._state.list_tab_ids() if key not in before]
                if len(added) != 1 or report.restored_tabs != 1:
                    raise RuntimeError(
                        "Workspace restore did not create exactly one tab"
                    )
                if snapshot.active_tab_index == index:
                    active = added[0]
        if active is not None:
            self._state.set_active_tab(active)
        self._skipped = tuple(skipped)
        return ReloadReport(restored_tabs=restored, rejected_tabs=tuple(issues))

    def retry_skipped_tabs(self) -> ReloadReport:
        self._require_idle()
        self._access.require_available()
        active = self._state.active_tab_id
        with self._switch():
            report = self._restore(PersistedSession(tabs=self._skipped))
            if active is not None and self._state.has_tab(active):
                self._state.set_active_tab(active)
            return report
