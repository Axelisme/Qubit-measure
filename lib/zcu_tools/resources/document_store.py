"""Typed optimistic transactions over an existing round-trip YAML document.

Each store owns an independent memory snapshot. ``snapshot`` never reads disk;
``edit`` reloads under short entry/commit locks and rejects nested transactions.
All changed fields merge or conflict as one transaction. Schema/custom validation
and sibling-file replacement precede memory publication. This is single-file
atomicity, not a crash journal or a multi-file durability guarantee.

Declare structural paths for known physical values and their stderr in ``units``.
Resolvers receive the current raw document. Only declared paths convert between
SI on disk and working units in the model; untouched YAML nodes remain intact.
Forward-minor fields stay outside the typed view without being removed on disk.

Observers run after publication and unlock. Their exceptions are logged at ERROR,
with traceback, rather than reclassifying a completed commit as failed. External
changes are adopted by ``refresh`` or a successful edit, including an empty edit;
this module does not run a file watcher.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from math import isnan
from pathlib import Path
from typing import Literal, cast

from filelock import FileLock, Timeout
from pydantic import BaseModel, TypeAdapter
from ruamel.yaml import YAML

from zcu_tools.format_version import FormatVersion, YamlMap, YamlValue, validate_header

from ._document_commit import DocumentEdit, FieldPath, PreparedDocument, stage_content

__all__ = (
    "ConflictError",
    "DocumentChange",
    "DocumentStore",
    "FieldPath",
    "LockTimeoutError",
    "UnitResolver",
    "UnitSpec",
)

_SUPPORTED_VERSION = FormatVersion(1, 0)


@dataclass(frozen=True)
class UnitSpec:
    """Units for one numeric scalar path, including a separate stderr path.

    si_unit is the stored unit; working_unit is the model/caller unit. Each is
    one of Hz, MHz, GHz, s, us, µs, A, mA, or 1 (dimensionless). Both must have
    the same dimension. For example, UnitSpec("Hz", "MHz") stores 1e6 for a
    working value of 1. Missing/null values stay missing/null; bool, sequences
    and other nonnumeric values are not physical scalars.
    """

    si_unit: str
    working_unit: str

    def validate(self) -> None:
        """Return None, or raise ValueError for unknown/incompatible unit names."""
        _si_per_working_unit(self)


type UnitResolver = Callable[[Mapping[str, YamlValue]], Mapping[FieldPath, UnitSpec]]


def _si_per_working_unit(spec: UnitSpec) -> float:
    """Return stored units per working unit; reject unsupported pairs with ValueError."""
    scales = {
        "Hz": ("Hz", 1.0),
        "MHz": ("Hz", 1e6),
        "GHz": ("Hz", 1e9),
        "s": ("s", 1.0),
        "us": ("s", 1e-6),
        "µs": ("s", 1e-6),
        "A": ("A", 1.0),
        "mA": ("A", 1e-3),
        "1": ("1", 1.0),
    }
    si = scales.get(spec.si_unit)
    working = scales.get(spec.working_unit)
    if si is None or working is None or si[0] != working[0]:
        raise ValueError(f"Incompatible or unsupported units: {spec!r}")
    return working[1] / si[1]


@dataclass(frozen=True)
class DocumentChange:
    """A published change delivered after the outermost store lock is released.

    source is the YAML file path. paths contains structural tuples of mapping
    keys, never split on dots; sequences count as a whole field. For reason
    "commit", paths come from changed typed model fields in working units.
    For reason "refresh", they come from raw SI YAML and can include header or
    forward-minor fields. A nonempty paths tuple reports structural differences,
    not every field present in the snapshot.
    """

    source: Path
    paths: tuple[FieldPath, ...]
    reason: Literal["commit", "refresh"]


class _Missing(Enum):
    # Absent keys must differ from YAML null during patch/conflict comparison.
    VALUE = "<missing>"


class ConflictError(RuntimeError):
    """A changed path no longer matches the raw SI baseline, so nothing commits.

    source is the YAML path; path is a tuple of unchanged mapping keys.
    original/current are the raw SI values at that path, or the absent-key
    marker displayed as "<missing>". Null is None, distinct from that marker.
    """

    def __init__(
        self,
        source: Path,
        path: FieldPath,
        original: YamlValue | _Missing,
        current: YamlValue | _Missing,
    ) -> None:
        """Record source/path and the conflicting raw original/current values."""
        self.source = source
        self.path = path
        self.original = original
        self.current = current
        super().__init__(
            f"{source}: conflict at {path!r}: original={original!r}, current={current!r}"
        )


class LockTimeoutError(TimeoutError):
    """Lock acquisition failed: lock_path is the sidecar file, timeout is seconds."""

    def __init__(self, lock_path: Path, timeout: float) -> None:
        """Record the attempted lock_path and wait limit; no document was changed."""
        self.lock_path = lock_path
        self.timeout = timeout
        super().__init__(f"{lock_path}: lock acquisition timed out after {timeout}s")


def _same_value(original: YamlValue | _Missing, current: YamlValue | _Missing) -> bool:
    if isinstance(original, dict) and isinstance(current, dict):
        return original.keys() == current.keys() and all(
            _same_value(value, current[key]) for key, value in original.items()
        )
    if isinstance(original, list) and isinstance(current, list):
        return len(original) == len(current) and all(
            _same_value(left, right)
            for left, right in zip(original, current, strict=True)
        )
    if isinstance(original, bool) != isinstance(current, bool):
        return False
    if (
        isinstance(original, float)
        and isinstance(current, float)
        and isnan(original)
        and isnan(current)
    ):
        return True
    return original == current


def _projection(value: object) -> YamlValue:
    """Serialize existing presence only; unset model fields stay absent.

    Recurse through explicitly serialized model fields, mappings and sequences.
    Reject the absent-key marker before leaf validation can coerce it to a string.
    """
    if isinstance(value, _Missing):
        raise TypeError("Missing values cannot be serialized to YAML")
    if isinstance(value, BaseModel):
        return {
            name: _projection(getattr(value, name))
            for name in value.model_dump(exclude_unset=True)
        }
    if isinstance(value, dict):
        value = {key: _projection(child) for key, child in value.items()}
    elif isinstance(value, list):
        value = [_projection(child) for child in value]
    return TypeAdapter(YamlValue).validate_python(value)


def _draft_model_projection(base: object, draft: BaseModel) -> YamlMap:
    """Include explicit fields and mutations of previously unset defaults.

    An existing model uses its edit-entry baseline. A new/replaced model uses
    declared defaults instead. Removing an explicitly present field keeps it
    absent. This model branch delegates nested containers to _draft_projection
    so defaults can themselves contain typed models.
    """
    fields = type(draft).model_fields
    values = {name: getattr(draft, name) for name in fields}
    same_model = isinstance(base, BaseModel) and type(base) is type(draft)
    base_fields = (
        base.model_fields_set if isinstance(base, BaseModel) and same_model else set()
    )
    candidate: YamlMap = {
        name: _draft_projection(
            getattr(base, name, _Missing.VALUE), getattr(draft, name)
        )
        for name in draft.model_dump(exclude_unset=True)
        if name not in fields
    }
    for name, field in fields.items():
        if same_model:
            before = getattr(base, name)
        elif field.is_required():
            before = _Missing.VALUE
        else:
            before = field.get_default(call_default_factory=True, validated_data=values)
        after = _draft_projection(before, values[name])
        if name in draft.model_fields_set or (
            name not in base_fields
            and (
                isinstance(before, _Missing)
                or not _same_value(_projection(before), after)
            )
        ):
            candidate[name] = after
    return candidate


def _draft_projection(base: object, draft: object) -> YamlValue:
    """Project only the candidate, preserving presence and in-place mutations.

    New mapping keys/list positions have no baseline; new models compare their
    unset fields with declared defaults. No unused original projection is built.
    """
    if isinstance(draft, BaseModel):
        return _draft_model_projection(base, draft)
    if isinstance(draft, dict):
        original = base if isinstance(base, dict) else {}
        return {
            key: _draft_projection(original.get(key, _Missing.VALUE), child)
            for key, child in draft.items()
        }
    if isinstance(draft, list):
        original_list = base if isinstance(base, list) else []
        return [
            _draft_projection(
                original_list[index] if index < len(original_list) else _Missing.VALUE,
                child,
            )
            for index, child in enumerate(draft)
        ]
    return _projection(draft)


def _changes(
    base: YamlValue | _Missing, draft: YamlValue | _Missing, path: FieldPath = ()
) -> Iterator[tuple[FieldPath, YamlValue | _Missing]]:
    """Yield changed mapping leaves; treat each list as one conflict/notice path."""
    if isinstance(base, dict) and isinstance(draft, dict):
        for key in dict.fromkeys((*base, *draft)):
            yield from _changes(
                base.get(key, _Missing.VALUE),
                draft.get(key, _Missing.VALUE),
                (*path, key),
            )
    elif not _same_value(base, draft):
        yield path, draft


def _patch_node(
    current_node: YamlValue | _Missing,
    base: YamlValue | _Missing,
    draft: YamlValue | _Missing,
) -> YamlValue | _Missing:
    """Preserve untouched SI nodes while merging working-unit replacements.

    The caller must first conflict-check changed paths against the original raw
    SI document. base/draft are typed working-unit projections, not raw nodes.
    The result mixes surviving SI nodes and changed working values until the
    caller converts changed paths back to SI. Lists conflict as a whole but
    surviving positions retain their round-trip nodes.
    """
    if _same_value(base, draft):
        return current_node
    if (
        isinstance(current_node, dict)
        and isinstance(base, dict)
        and isinstance(draft, dict)
    ):
        for key in dict.fromkeys((*base, *draft)):
            value = _patch_node(
                current_node.get(key, _Missing.VALUE),
                base.get(key, _Missing.VALUE),
                draft.get(key, _Missing.VALUE),
            )
            if isinstance(value, _Missing):
                current_node.pop(key, None)
            else:
                current_node[key] = value
        return current_node
    if (
        isinstance(current_node, list)
        and isinstance(base, list)
        and isinstance(draft, list)
    ):
        # A sequence conflicts as one field, but surviving positions keep raw nodes.
        for index, value in enumerate(draft):
            if index < len(base) and index < len(current_node):
                merged = _patch_node(current_node[index], base[index], value)
                if not isinstance(merged, _Missing):
                    current_node[index] = merged
            else:
                current_node.append(value)
        del current_node[len(draft) :]
        return current_node
    return draft


def _lookup(document: YamlValue, path: FieldPath) -> YamlValue | _Missing:
    value: YamlValue | _Missing = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return _Missing.VALUE
        value = value[key]
    return value


def _apply(document: YamlMap, path: FieldPath, value: YamlValue | _Missing) -> None:
    parent = document
    for key in path[:-1]:
        child = parent.get(key)
        if not isinstance(child, dict):
            child = {}
            parent[key] = child
        parent = child
    if isinstance(value, _Missing):
        parent.pop(path[-1], None)
    else:
        parent[path[-1]] = value


class DocumentStore[T: BaseModel]:
    """An independent typed snapshot and optimistic transactions for one YAML file.

    model T owns the document shape; format identifies its persisted artifact.
    This store owns unit conversion, raw-node preservation and single-file
    replacement. It never creates missing documents or watches for changes.
    A snapshot is memory-only; edit/refresh reload disk. Observers see published
    state after unlock and cannot roll back a completed write.
    """

    def __init__(  # noqa: PLR0913 -- The accepted Interface fixes these arguments.
        self,
        path: Path,
        model: type[T],
        *,
        format: str,
        supported_version: FormatVersion = _SUPPORTED_VERSION,
        units: Mapping[FieldPath, UnitSpec] | UnitResolver | None = None,
        validate: Callable[[T], None] | None = None,
        lock_path: Path | None = None,
        lock_timeout: float = 10.0,
    ) -> None:
        """Open an existing UTF-8 YAML mapping and validate its initial snapshot.

        path names the document, not its directory. model is the BaseModel class
        for its full shape, including format and format_version fields. format
        is the exact artifact identifier in the YAML header, e.g.
        "zcu.parameter-container" for setup/point, not a kind name or "yaml".
        supported_version sets the accepted major/current minor. Future minor
        fields survive raw round-trip but stay outside the typed snapshot.

        units maps structural key tuples to UnitSpec, or resolves that mapping
        from the current document. None disables conversion. Declare value and
        stderr paths separately. Resolvers choose paths from structure or
        discriminators; during commit, changed scalars are still in working
        units while untouched nodes remain SI. validate receives each converted
        typed model after schema validation and may raise to reject it.

        lock_path selects a shared sidecar, defaulting to path + ".lock".
        lock_timeout is FileLock's wait in seconds (negative waits indefinitely).
        Stores sharing a document must use the same sidecar. This lock does not
        synchronize threads that share a store; each handle owns its state.

        Missing/unreadable files, YAML/header/version/schema errors, invalid
        physical values and validate exceptions propagate without publication.
        Initialization reads without acquiring the sidecar, allowing creation
        under a caller-held lock. Writers must replace atomically, as edit does.
        """
        self._path = path
        self._model = model
        self._format = format
        self._supported_version = supported_version
        self._units = units
        self._validate = validate
        self._lock_path = lock_path or Path(f"{path}.lock")
        self._lock_timeout = lock_timeout
        self._lock = FileLock(str(self._lock_path))
        self._editing = False
        self._observers: dict[int, Callable[[DocumentChange], None]] = {}
        self._next_subscription_id = 0
        self._deferred_until_unlock: list[DocumentChange] = []
        self._document, self._snapshot = self._read()

    def snapshot(self) -> T:
        """Return a deep independent copy in working units, without I/O or notice."""
        return self._snapshot.model_copy(deep=True)

    @contextmanager
    def edit(self) -> Generator[T]:
        """Yield an independent working-unit draft, then commit one atomic edit.

        Entry and commit each briefly lock/reload; the body runs unlocked unless
        the caller holds locked(). Changed paths compare against entry-time raw
        SI values. A conflict rejects all changes; body/schema/validate/I/O errors
        do not publish a draft. Nested edits on this store raise RuntimeError.
        Missing, null, deleted and in-place default changes remain distinct.

        A successful empty edit adopts the latest disk snapshot without rewriting
        or notifying. Nonempty edits notify after publication and outermost unlock;
        observer exceptions are logged, not raised as commit failures.
        """
        prepared: PreparedDocument[T] | None = None
        with ExitStack() as stack:
            with self.locked():
                state = stack.enter_context(self.read_state(locked_by=self))
            yield state.draft
            try:
                with self.locked():
                    prepared = self.prepare(state, locked_by=self)
                    prepared.replace()
                    self.publish(prepared, locked_by=self)
            finally:
                if prepared is not None:
                    prepared.discard()
        if prepared is not None:
            self.notify_commit(prepared)

    @contextmanager
    def read_state[_Owner: BaseModel](
        self, *, locked_by: DocumentStore[_Owner]
    ) -> Generator[DocumentEdit[T]]:
        """Yield raw SI base, typed working base and an independent working draft.

        Resources-only seam: locked_by must hold the same sidecar path. Reload
        and reject a nested edit; retain edit ownership until this context exits.
        This method never writes or publishes, and its caller controls lock scope.
        """
        self._require_lock(locked_by)
        if self._editing:
            raise RuntimeError(f"{self._path}: nested edits are not allowed")
        self._editing = True
        try:
            base_document, base = self._read()
            yield DocumentEdit(base_document, base, base.model_copy(deep=True))
        finally:
            self._editing = False

    def prepare[_Owner: BaseModel](
        self, state: DocumentEdit[T], *, locked_by: DocumentStore[_Owner]
    ) -> PreparedDocument[T]:
        """Prepare current SI YAML and a validated working snapshot without publishing.

        Resources-only seam: locked_by must hold the same sidecar path. Compare
        state base/draft in working units, reject raw SI conflicts, merge and
        convert changed paths to SI, then validate. Stage a sibling temporary
        only for nonempty patches. Caller must discard it even if replacement or
        publication fails; no disk replacement has happened on return.
        """
        self._require_lock(locked_by)
        base_values = _projection(state.base)
        draft_values = _draft_projection(state.base, state.draft)
        patches = tuple(_changes(base_values, draft_values))
        document, _ = self._read()
        original = self._path.read_bytes()
        self._merge_patches(document, state.base_document, base_values, patches)
        self._to_si_units(document, tuple(path for path, _ in patches))
        snapshot = self._model_snapshot(document)
        temporary: Path | None = None
        if patches:
            stream = StringIO()
            YAML(typ="rt").dump(document, stream)
            temporary = stage_content(self._path, stream.getvalue().encode("utf-8"))
        return PreparedDocument(
            self._path,
            document,
            snapshot,
            tuple(path for path, _ in patches),
            original,
            temporary,
        )

    def publish[_Owner: BaseModel](
        self, prepared: PreparedDocument[T], *, locked_by: DocumentStore[_Owner]
    ) -> None:
        """Adopt prepared SI YAML/working snapshot after its successful replacement.

        Resources-only seam: locked_by holds the same sidecar; reject a prepared
        document for another source. No I/O or notification occurs here.
        """
        self._require_lock(locked_by)
        if prepared.source != self._path:
            raise ValueError(
                f"{self._path}: prepared document belongs to {prepared.source}"
            )
        self._snapshot = prepared.snapshot
        self._document = prepared.document

    def notify_commit(self, prepared: PreparedDocument[T]) -> None:
        """Dispatch a published nonempty edit; queue if an outer lock is still held."""
        if prepared.paths:
            self._dispatch_change(DocumentChange(self._path, prepared.paths, "commit"))

    def _require_lock[_Owner: BaseModel](self, owner: DocumentStore[_Owner]) -> None:
        if self._lock_path != owner._lock_path or not owner._lock.is_locked:
            raise RuntimeError(
                f"{self._path}: commit seam requires the matching held lock"
            )

    def _merge_patches(
        self,
        document: YamlMap,
        base_document: YamlMap,
        base_values: YamlValue,
        patches: tuple[tuple[FieldPath, YamlValue | _Missing], ...],
    ) -> None:
        """Conflict-check raw SI baselines, then merge working-unit patches in place.

        document/base_document are current/original raw SI trees; base_values
        and patch replacements are typed working-unit projections. All conflicts
        are checked before mutation. The result retains untouched raw SI nodes;
        changed nodes remain working units for _to_si_units.
        """
        for path, _ in patches:
            self._check_conflict(base_document, document, path)
        for path, value in patches:
            merged = _patch_node(
                _lookup(document, path), _lookup(base_values, path), value
            )
            _apply(document, path, merged)

    def _to_si_units(
        self, document: YamlMap, changed_paths: tuple[FieldPath, ...]
    ) -> None:
        """Convert only changed working-unit paths in a merged tree to SI in place.

        Resolvers see the merged structure/discriminators before conversion.
        Unchanged raw SI nodes and undeclared extension values remain untouched.
        Unsupported units or nonnumeric physical scalars raise ValueError.
        """
        for path, spec in self._unit_specs(document).items():
            if any(path[: len(changed)] == changed for changed in changed_paths):
                value = self._scale_value(
                    _lookup(document, path), _si_per_working_unit(spec), path
                )
                if not isinstance(value, _Missing):
                    _apply(document, path, value)

    def _check_conflict(self, base: YamlMap, current: YamlMap, path: FieldPath) -> None:
        for length in range(1, len(path) + 1):
            prefix = path[:length]
            original = _lookup(base, prefix)
            latest = _lookup(current, prefix)
            # Sibling edits may coexist; ancestors must remain mappings.
            if (
                length < len(path)
                and isinstance(original, dict)
                and isinstance(latest, dict)
            ):
                continue
            if not _same_value(original, latest):
                raise ConflictError(self._path, prefix, original, latest)

    def _read(self) -> tuple[YamlMap, T]:
        """Read quote-preserving SI YAML and validate a converted working snapshot.

        No lock is acquired here. Edit/refresh callers own their lock scope;
        initial construction may run while a different owner already holds it.
        """
        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        with self._path.open(encoding="utf-8") as stream:
            raw = yaml.load(stream)
        # Validate the recursive shape without discarding ruamel's round-trip nodes.
        TypeAdapter(YamlMap).validate_python(raw, strict=True)
        document = cast(YamlMap, raw)
        return document, self._model_snapshot(document)

    def _model_snapshot(self, document: YamlMap) -> T:
        """Check the header, convert SI to working units, validate T and custom rules.

        Future-minor fields are ignored only in the typed result. Validation
        failures propagate before callers can publish it or replace the file.
        """
        version = validate_header(
            document,
            expected_format=self._format,
            supported_version=self._supported_version,
            source=self._path,
        )
        # Ignore future fields only in the typed view; retain them in the YAML tree.
        extra = "ignore" if version.minor > self._supported_version.minor else None
        snapshot = self._model.model_validate(
            self._working_values(document), extra=extra
        )
        if self._validate is not None:
            self._validate(snapshot)
        return snapshot

    def _unit_specs(self, document: YamlMap) -> Mapping[FieldPath, UnitSpec]:
        """Resolve paths from current structure, possibly before changed SI conversion."""
        return self._units(document) if callable(self._units) else self._units or {}

    def _working_values(self, document: YamlMap) -> YamlMap:
        """Copy raw SI YAML to model units; keep null/missing and undeclared values."""
        values = TypeAdapter(YamlMap).validate_python(document)
        for path, spec in self._unit_specs(document).items():
            value = self._scale_value(
                _lookup(values, path), 1 / _si_per_working_unit(spec), path
            )
            if not isinstance(value, _Missing):
                _apply(values, path, value)
        return values

    def _scale_value(
        self, value: YamlValue | _Missing, factor: float, path: FieldPath
    ) -> YamlValue | _Missing:
        if value is None or isinstance(value, _Missing):
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{self._path}: {path!r} must be a numeric physical value")
        return value * factor

    def refresh(self) -> bool:
        """Reload/validate disk and publish its working-unit snapshot.

        Return True and notify if raw SI YAML differs, otherwise return False.
        Header/version/schema/validate/I/O and lock failures propagate and leave
        the prior snapshot intact. Notifications wait for outermost unlock;
        observer failures are logged separately from successful publication.
        """
        with self.locked():
            document, snapshot = self._read()
            paths = tuple(path for path, _ in _changes(self._document, document))
            self._snapshot = snapshot
            self._document = document
        if paths:
            self._dispatch_change(DocumentChange(self._path, paths, "refresh"))
        return bool(paths)

    @contextmanager
    def locked(self) -> Generator[None]:
        """Hold this store's real reentrant sidecar lock until context exit.

        Raise LockTimeoutError when acquisition exceeds lock_timeout. An outer
        locked() also holds through edit bodies and defers commit/refresh notices.
        The outermost exit releases the lock before delivering queued observers,
        including on exceptional exit; nested exits do not deliver them.
        """
        try:
            self._lock.acquire(timeout=self._lock_timeout)
        except Timeout as exc:
            raise LockTimeoutError(self._lock_path, self._lock_timeout) from exc
        try:
            yield
        finally:
            self._lock.release()
            if not self._lock.is_locked:
                deferred, self._deferred_until_unlock = self._deferred_until_unlock, []
                for change in deferred:
                    self._notify(change)

    def subscribe(
        self, callback: Callable[[DocumentChange], None]
    ) -> Callable[[], None]:
        """Register callback(change) and return an idempotent unsubscribe function.

        Every registration is independent, including duplicate callbacks.
        Delivery uses a registration-order snapshot after publication/unlock;
        callback exceptions are logged with traceback and do not undo a commit.
        """
        subscription_id = self._next_subscription_id
        self._next_subscription_id += 1
        self._observers[subscription_id] = callback

        def unsubscribe() -> None:
            self._observers.pop(subscription_id, None)

        return unsubscribe

    def _dispatch_change(self, change: DocumentChange) -> None:
        """Notify now, or queue until locked() releases the outermost reentrant lock."""
        if self._lock.is_locked:
            self._deferred_until_unlock.append(change)
        else:
            self._notify(change)

    def _notify(self, change: DocumentChange) -> None:
        for callback in tuple(self._observers.values()):
            try:
                callback(change)
            except Exception:
                # Notification failure cannot roll back a published transaction.
                logging.getLogger(__name__).exception(
                    "%s: observer failed after %s publication",
                    self._path,
                    change.reason,
                )
