"""Creation and identity of a result entry across two explicit roots."""

import errno
import os
import shutil
from collections.abc import Generator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from ruamel.yaml import YAML

from zcu_tools.format_version import YamlValue
from zcu_tools.resources._document_commit import PreparedDocument
from zcu_tools.resources.document_store import (
    DocumentStore,
    FieldPath,
    UnitSpec,
)

from . import _point_origin
from .errors import PartialCommitError
from .layering import compose, point_units, route, validate_point
from .points import PointView
from .registry import component_registry
from .schema import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    LayeredDocument,
    PointDocument,
    SetupDocument,
    is_forward_minor,
    validate_component_name,
)
from .views import SetupView


def _entry_path(root: str | Path, name: str, *, new_destination: bool = False) -> Path:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or Path(name).is_absolute()
        or PureWindowsPath(name).anchor
    ):
        raise ValueError(f"{name!r}: expected a single path component")
    root_path = Path(root)
    path = root_path / name
    if new_destination and (path.exists() or path.is_symlink()):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))
    if path.resolve().parent != root_path.resolve():
        raise ValueError(f"{path}: entry escapes its root")
    return path


def rename_entry(
    old: str, new: str, *, result_root: str | Path, database_root: str | Path
) -> None:
    result_old = _entry_path(result_root, old)
    database_old = _entry_path(database_root, old)
    result_new = _entry_path(result_root, new, new_destination=True)
    database_new = _entry_path(database_root, new, new_destination=True)
    ResultEntry.open(old, result_root=result_root, database_root=database_root)
    result_old.rename(result_new)
    try:
        database_old.rename(database_new)
    except OSError as cause:
        try:
            if result_old.exists() or result_old.is_symlink():
                raise FileExistsError(
                    errno.EEXIST, os.strerror(errno.EEXIST), str(result_old)
                )
            result_new.rename(result_old)
        except OSError as recovery_cause:
            raise PartialCommitError(
                completed=(result_new,),
                pending=(database_new,),
                recovery_failed=(result_old,),
                cause=cause,
                recovery_cause=recovery_cause,
            ) from cause
        raise


@dataclass
class _PointContext:
    setup: SetupDocument


class ResultEntry:
    def __init__(self, result_path: Path, database_path: Path) -> None:
        self._result_path = result_path
        self._database_path = database_path
        self._entry_id: str | None = None
        source = result_path / "setup.yaml"

        # Bind diagnostics to this handle without shared mutable model context.
        class EntrySetupDocument(SetupDocument):
            _source = source

        self._setup_store = DocumentStore[SetupDocument](
            source,
            EntrySetupDocument,
            format=PARAMETER_FORMAT,
            supported_version=PARAMETER_VERSION,
            units=self._setup_units,
            validate=self._validate_setup,
            lock_path=result_path / ".entry.lock",
        )

    def _setup_units(
        self, document: Mapping[str, YamlValue]
    ) -> Mapping[FieldPath, UnitSpec]:
        result: dict[FieldPath, UnitSpec] = {}
        forward_minor = is_forward_minor(
            document, source=self._result_path / "setup.yaml"
        )
        components = document.get("components")
        if isinstance(components, dict):
            for name, fields in components.items():
                validate_component_name(name, source=self._result_path / "setup.yaml")
                if isinstance(fields, dict) and isinstance(
                    kind := fields.get("kind"), str
                ):
                    component_registry.get(
                        kind, source=self._result_path / "setup.yaml", component=name
                    )
                    if not forward_minor:
                        component_registry.check_fields(kind, fields, path=name)
                    for path, spec in component_registry.units(
                        kind, source=self._result_path / "setup.yaml", component=name
                    ).items():
                        result[("components", name, *path)] = spec
        return result

    def _validate_setup(self, document: SetupDocument) -> None:
        component_registry.validate_references(
            document.components, source=self._result_path / "setup.yaml"
        )
        if self._entry_id is None:
            self._entry_id = document.general.entry_id
        elif document.general.entry_id != self._entry_id:
            raise ValueError(
                f"{self._result_path / 'setup.yaml'}: entry_id is immutable; "
                f"expected {self._entry_id!r}, got {document.general.entry_id!r}"
            )

    @property
    def setup(self) -> SetupView:
        return SetupView(
            self._setup_store, self._result_path / "setup.yaml", self._edit_setup
        )

    @property
    def entry_id(self) -> str:
        return self._setup_store.snapshot().general.entry_id

    def list_points(self) -> list[str]:
        return sorted(
            path.name
            for path in (self._result_path / "points").iterdir()
            if path.is_dir()
            and (path / "point.yaml").is_file()
            and (path / "module_cfg.yaml").is_file()
        )

    def new_point(
        self, label: str, *, clone_from: str | PointView | None = None
    ) -> PointView:
        destination = _entry_path(
            self._result_path / "points", label, new_destination=True
        )
        if clone_from is not None:
            destination.mkdir()
            try:
                self._clone_point(destination, clone_from)
                return self.use_point(label)
            except BaseException:
                shutil.rmtree(destination)
                raise
        destination.mkdir()
        try:
            document = {
                "format": PARAMETER_FORMAT,
                "format_version": f"{PARAMETER_VERSION.major}.{PARAMETER_VERSION.minor}",
                "general": {
                    "created_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                },
                "components": {},
                "provenance": {},
            }
            with (destination / "point.yaml").open("x", encoding="utf-8") as stream:
                YAML(typ="rt").dump(document, stream)
            with (destination / "module_cfg.yaml").open(
                "x", encoding="utf-8"
            ) as stream:
                YAML(typ="rt").dump(
                    {"format": "zcu.module-library", "format_version": "1.0"}, stream
                )
            return self.use_point(label)
        except BaseException:
            shutil.rmtree(destination)
            raise

    def _clone_point(self, destination: Path, clone_from: str | PointView) -> None:
        with ExitStack() as stack, self._setup_store.locked():
            source = (
                _point_origin.clone_source(clone_from, self._result_path)
                if isinstance(clone_from, PointView)
                else _entry_path(self._result_path / "points", clone_from)
            )
            setup_state = stack.enter_context(
                self._setup_store.read_state(locked_by=self._setup_store)
            )
            store, _ = self._point_store(source / "point.yaml", setup_state.base)
            compose(
                setup_state.base,
                store.snapshot(),
                source / "point.yaml",
                complete=True,
            )
            for filename in ("point.yaml", "module_cfg.yaml"):
                shutil.copyfile(source / filename, destination / filename)
            cloned_store, _ = self._point_store(
                destination / "point.yaml", setup_state.base
            )
            cloned_state = stack.enter_context(
                cloned_store.read_state(locked_by=self._setup_store)
            )
            cloned_state.draft.general.created_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            for metadata in cloned_state.draft.provenance.values():
                metadata["cloned_from"] = {
                    "entry_id": setup_state.base.general.entry_id,
                    "point": source.name,
                }
            prepared = cloned_store.prepare(cloned_state, locked_by=self._setup_store)
            try:
                prepared.replace()
            finally:
                prepared.discard()

    def use_point(self, label: str) -> PointView:
        directory = _entry_path(self._result_path / "points", label)
        source = directory / "point.yaml"
        module = directory / "module_cfg.yaml"
        if not module.is_file():
            raise FileNotFoundError(module)
        with ExitStack() as stack, self._setup_store.locked():
            setup_state = stack.enter_context(
                self._setup_store.read_state(locked_by=self._setup_store)
            )
            setup_prepared = self._setup_store.prepare(
                setup_state, locked_by=self._setup_store
            )
            store, context = self._point_store(source, setup_prepared.snapshot)
            compose(context.setup, store.snapshot(), source, complete=True)
            self._setup_store.publish(setup_prepared, locked_by=self._setup_store)
        return PointView(
            store,
            self._setup_store,
            source,
            lambda: self._edit_point(store, context, source),
            lambda: self._refresh_point(store, context, source),
        )

    def _refresh_point(
        self,
        store: DocumentStore[PointDocument],
        context: _PointContext,
        source: Path,
    ) -> None:
        with ExitStack() as stack, self._setup_store.locked():
            setup_state = stack.enter_context(
                self._setup_store.read_state(locked_by=self._setup_store)
            )
            setup_prepared = self._setup_store.prepare(
                setup_state, locked_by=self._setup_store
            )
            context.setup = setup_prepared.snapshot
            try:
                point_state = stack.enter_context(
                    store.read_state(locked_by=self._setup_store)
                )
                point_prepared = store.prepare(point_state, locked_by=self._setup_store)
                compose(
                    setup_prepared.snapshot,
                    point_prepared.snapshot,
                    source,
                    complete=True,
                )
                self._setup_store.publish(setup_prepared, locked_by=self._setup_store)
                store.publish(point_prepared, locked_by=self._setup_store)
            finally:
                context.setup = self._setup_store.snapshot()

    def _point_store(
        self, source: Path, setup: SetupDocument
    ) -> tuple[DocumentStore[PointDocument], _PointContext]:
        context = _PointContext(setup)
        store = DocumentStore[PointDocument](
            source,
            PointDocument,
            format=PARAMETER_FORMAT,
            supported_version=PARAMETER_VERSION,
            units=lambda document: point_units(document, context.setup, source),
            validate=lambda document: validate_point(document, context.setup, source),
            lock_path=self._result_path / ".entry.lock",
        )
        return store, context

    def _validate_all_points(
        self,
        setup: SetupDocument,
        *,
        override: tuple[Path, PointDocument] | None = None,
    ) -> None:
        for label in self.list_points():
            source = self._result_path / "points" / label / "point.yaml"
            if override is not None and source == override[0]:
                point = override[1]
            else:
                store, _ = self._point_store(source, setup)
                point = store.snapshot()
            compose(setup, point, source, complete=True)

    @contextmanager
    def _edit_setup(self) -> Generator[SetupDocument]:
        prepared: PreparedDocument[SetupDocument] | None = None
        with ExitStack() as stack:
            with self._setup_store.locked():
                state = stack.enter_context(
                    self._setup_store.read_state(locked_by=self._setup_store)
                )
            yield state.draft
            try:
                with self._setup_store.locked():
                    prepared = self._setup_store.prepare(
                        state, locked_by=self._setup_store
                    )
                    self._validate_all_points(prepared.snapshot)
                    prepared.replace()
                    self._setup_store.publish(prepared, locked_by=self._setup_store)
            finally:
                if prepared is not None:
                    prepared.discard()
        if prepared is not None:
            self._setup_store.notify_commit(prepared)

    @contextmanager
    def _edit_point(
        self,
        store: DocumentStore[PointDocument],
        context: _PointContext,
        source: Path,
    ) -> Generator[LayeredDocument]:
        setup_prepared: PreparedDocument[SetupDocument] | None = None
        point_prepared: PreparedDocument[PointDocument] | None = None
        with ExitStack() as stack:
            with self._setup_store.locked():
                # D109 package-internal seams; entry owns the one shared lock.
                setup_state = stack.enter_context(
                    self._setup_store.read_state(locked_by=self._setup_store)
                )
                context.setup = setup_state.base
                point_state = stack.enter_context(
                    store.read_state(locked_by=self._setup_store)
                )
            compose(setup_state.base, point_state.base, source, complete=True)
            before = compose(setup_state.base, point_state.base, source, complete=False)
            draft = before.model_copy(deep=True)
            yield draft
            route(before, draft, setup_state.draft, point_state.draft)
            try:
                with self._setup_store.locked():
                    setup_prepared = self._setup_store.prepare(
                        setup_state, locked_by=self._setup_store
                    )
                    context.setup = setup_prepared.snapshot
                    point_prepared = store.prepare(
                        point_state, locked_by=self._setup_store
                    )
                    self._validate_all_points(
                        setup_prepared.snapshot,
                        override=(source, point_prepared.snapshot),
                    )
                    self._replace_layers((setup_prepared, point_prepared))
                    self._setup_store.publish(
                        setup_prepared, locked_by=self._setup_store
                    )
                    store.publish(point_prepared, locked_by=self._setup_store)
            finally:
                if setup_prepared is not None:
                    setup_prepared.discard()
                if point_prepared is not None:
                    point_prepared.discard()
        if setup_prepared is not None:
            self._setup_store.notify_commit(setup_prepared)
        if point_prepared is not None:
            store.notify_commit(point_prepared)

    @staticmethod
    def _replace_layers(
        layers: tuple[
            PreparedDocument[SetupDocument] | PreparedDocument[PointDocument], ...
        ],
    ) -> None:
        replaced: list[
            PreparedDocument[SetupDocument] | PreparedDocument[PointDocument]
        ] = []
        try:
            for layer in layers:
                if layer.temporary is not None:
                    layer.replace()
                    replaced.append(layer)
        except OSError as cause:
            recovery_failed: list[Path] = []
            recovery_cause: OSError | None = None
            for layer in reversed(replaced):
                try:
                    layer.restore()
                except OSError as error:
                    recovery_failed.append(layer.source)
                    recovery_cause = error
            if recovery_cause is not None:
                raise PartialCommitError(
                    completed=tuple(recovery_failed),
                    pending=tuple(
                        layer.source
                        for layer in layers
                        if layer.source not in recovery_failed
                    ),
                    recovery_failed=tuple(recovery_failed),
                    cause=cause,
                    recovery_cause=recovery_cause,
                ) from recovery_cause
            raise

    @classmethod
    def create(
        cls, name: str, *, result_root: str | Path, database_root: str | Path
    ) -> "ResultEntry":
        result_path = _entry_path(result_root, name, new_destination=True)
        database_path = _entry_path(database_root, name, new_destination=True)
        created: list[Path] = []
        try:
            for destination in (result_path, database_path):
                destination.mkdir(parents=True)
                created.append(destination)
            (result_path / "records").mkdir()
            (result_path / "points").mkdir()
            document = {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {
                    "entry_id": str(uuid4()),
                    "created_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                },
                "components": {},
                "provenance": {},
            }
            with (result_path / "setup.yaml").open("x", encoding="utf-8") as stream:
                YAML(typ="rt").dump(document, stream)
            return cls(result_path, database_path)
        except BaseException:
            # These entry directories belong to this call, not to the caller's roots.
            for directory in reversed(created):
                shutil.rmtree(directory)
            raise

    @classmethod
    def open(
        cls, name: str, *, result_root: str | Path, database_root: str | Path
    ) -> "ResultEntry":
        result_path = _entry_path(result_root, name)
        database_path = _entry_path(database_root, name)
        for directory in (
            result_path,
            database_path,
            result_path / "points",
            result_path / "records",
        ):
            if not directory.exists():
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), str(directory)
                )
            if not directory.is_dir():
                raise NotADirectoryError(
                    errno.ENOTDIR, os.strerror(errno.ENOTDIR), str(directory)
                )
        return cls(result_path, database_path)
