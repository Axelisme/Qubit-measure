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
changes require ``refresh``; this module does not run a file watcher.
"""

import logging
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from math import isnan
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, cast

from filelock import FileLock, Timeout
from pydantic import BaseModel, TypeAdapter
from ruamel.yaml import YAML

from zcu_tools.format_version import FormatVersion, YamlMap, YamlValue, validate_header

type FieldPath = tuple[str, ...]

_SUPPORTED_VERSION = FormatVersion(1, 0)


@dataclass(frozen=True)
class UnitSpec:
    si_unit: str
    working_unit: str

    def validate(self) -> None:
        """Reject units unsupported by the DocumentStore conversion boundary."""
        _unit_factor(self)


type UnitResolver = Callable[[Mapping[str, YamlValue]], Mapping[FieldPath, UnitSpec]]


def _unit_factor(spec: UnitSpec) -> float:
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
    source: Path
    paths: tuple[FieldPath, ...]
    reason: Literal["commit", "refresh"]


class _Missing(Enum):
    VALUE = "<missing>"


class ConflictError(RuntimeError):
    def __init__(
        self,
        source: Path,
        path: FieldPath,
        original: YamlValue | _Missing,
        current: YamlValue | _Missing,
    ) -> None:
        self.source = source
        self.path = path
        self.original = original
        self.current = current
        super().__init__(
            f"{source}: conflict at {path!r}: original={original!r}, current={current!r}"
        )


class LockTimeoutError(TimeoutError):
    def __init__(self, lock_path: Path, timeout: float) -> None:
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


def _yaml_value(value: object) -> YamlValue:
    if isinstance(value, BaseModel):
        result = TypeAdapter(YamlMap).validate_python(
            value.model_dump(exclude_unset=True)
        )
        values = {name: getattr(value, name) for name in type(value).model_fields}
        for name, field in type(value).model_fields.items():
            if name in result:
                result[name] = _yaml_value(values[name])
            elif not field.is_required():
                # New children have no edit-entry baseline. Compare declared
                # defaults recursively, retaining nested presence and mutations.
                default = field.get_default(
                    call_default_factory=True, validated_data=values
                )
                before, after = _edit_values(default, values[name])
                if not _same_value(before, after):
                    result[name] = after
        return result
    if isinstance(value, dict):
        value = {key: _yaml_value(child) for key, child in value.items()}
    elif isinstance(value, list):
        value = [_yaml_value(child) for child in value]
    return TypeAdapter(YamlValue).validate_python(value)


def _edit_values(base: object, draft: object) -> tuple[YamlValue, YamlValue]:
    if (
        isinstance(base, BaseModel)
        and isinstance(draft, BaseModel)
        and type(base) is type(draft)
    ):
        original = TypeAdapter(YamlMap).validate_python(
            base.model_dump(exclude_unset=True)
        )
        candidate = TypeAdapter(YamlMap).validate_python(
            draft.model_dump(exclude_unset=True)
        )
        for name in type(draft).model_fields:
            before, after = _edit_values(getattr(base, name), getattr(draft, name))
            if name in original:
                original[name] = before
            # An in-place edit does not mark its containing field as explicitly set.
            if name in candidate or not _same_value(before, after):
                candidate[name] = after
        return original, candidate
    if isinstance(base, dict) and isinstance(draft, dict):
        original_map = cast(dict[str, object], base)
        draft_map = cast(dict[str, object], draft)
        return (
            {key: _edit_values(child, child)[0] for key, child in original_map.items()},
            {
                key: _edit_values(original_map.get(key, _Missing.VALUE), child)[1]
                for key, child in draft_map.items()
            },
        )
    if isinstance(base, list) and isinstance(draft, list):
        original_list = cast(list[object], base)
        draft_list = cast(list[object], draft)
        return (
            [_edit_values(child, child)[0] for child in original_list],
            [
                _edit_values(
                    original_list[index]
                    if index < len(original_list)
                    else _Missing.VALUE,
                    child,
                )[1]
                for index, child in enumerate(draft_list)
            ],
        )
    return _yaml_value(base), _yaml_value(draft)


def _changes(
    base: YamlValue | _Missing, draft: YamlValue | _Missing, path: FieldPath = ()
) -> Iterator[tuple[FieldPath, YamlValue | _Missing]]:
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
    raw: YamlValue | _Missing, base: YamlValue | _Missing, draft: YamlValue | _Missing
) -> YamlValue | _Missing:
    if _same_value(base, draft):
        return raw
    if isinstance(raw, dict) and isinstance(base, dict) and isinstance(draft, dict):
        for key in dict.fromkeys((*base, *draft)):
            value = _patch_node(
                raw.get(key, _Missing.VALUE),
                base.get(key, _Missing.VALUE),
                draft.get(key, _Missing.VALUE),
            )
            if isinstance(value, _Missing):
                raw.pop(key, None)
            else:
                raw[key] = value
        return raw
    if isinstance(raw, list) and isinstance(base, list) and isinstance(draft, list):
        # A sequence conflicts as one field, but its surviving positions keep raw nodes.
        for index, value in enumerate(draft):
            if index < len(base) and index < len(raw):
                merged = _patch_node(raw[index], base[index], value)
                if not isinstance(merged, _Missing):
                    raw[index] = merged
            else:
                raw.append(value)
        del raw[len(draft) :]
        return raw
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
        self._observers: dict[object, Callable[[DocumentChange], None]] = {}
        self._pending_changes: list[DocumentChange] = []
        yaml = YAML(typ="rt")
        with path.open(encoding="utf-8") as stream:
            document = TypeAdapter(YamlMap).validate_python(yaml.load(stream))
        self._snapshot = self._model_snapshot(document)
        self._document = document

    def snapshot(self) -> T:
        return self._snapshot.model_copy(deep=True)

    @contextmanager
    def edit(self) -> Generator[T]:
        if self._editing:
            raise RuntimeError(f"{self._path}: nested edits are not allowed")
        self._editing = True
        try:
            with self.locked():
                base_document, base = self._read()
                draft = base.model_copy(deep=True)
            yield draft
            base_values, draft_values = _edit_values(base, draft)
            patches = tuple(_changes(base_values, draft_values))
            with self.locked():
                document, _ = self._read()
                for path, _ in patches:
                    self._check_conflict(base_document, document, path)
                for path, value in patches:
                    merged = _patch_node(
                        _lookup(document, path), _lookup(base_values, path), value
                    )
                    _apply(document, path, merged)
                # Resolve after structural edits; changed values are still in working units.
                for path, spec in self._unit_specs(document).items():
                    if any(path[: len(changed)] == changed for changed, _ in patches):
                        value = self._scale_value(
                            _lookup(document, path), _unit_factor(spec), path
                        )
                        if not isinstance(value, _Missing):
                            _apply(document, path, value)
                snapshot = self._model_snapshot(document)
                if patches:
                    self._write(document)
                self._snapshot = snapshot
                self._document = document
        finally:
            self._editing = False
        if patches:
            self._dispatch_change(
                DocumentChange(self._path, tuple(path for path, _ in patches), "commit")
            )

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
        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        with self._path.open(encoding="utf-8") as stream:
            raw = yaml.load(stream)
        # Validate the recursive shape without discarding ruamel's round-trip nodes.
        TypeAdapter(YamlMap).validate_python(raw, strict=True)
        document = cast(YamlMap, raw)
        return document, self._model_snapshot(document)

    def _model_snapshot(self, document: YamlMap) -> T:
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
        return self._units(document) if callable(self._units) else self._units or {}

    def _working_values(self, document: YamlMap) -> YamlMap:
        values = TypeAdapter(YamlMap).validate_python(document)
        for path, spec in self._unit_specs(document).items():
            value = self._scale_value(
                _lookup(values, path), 1 / _unit_factor(spec), path
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

    def _write(self, document: YamlMap) -> None:
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                YAML(typ="rt").dump(document, stream)
            temporary.replace(self._path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def refresh(self) -> bool:
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
        try:
            self._lock.acquire(timeout=self._lock_timeout)
        except Timeout as exc:
            raise LockTimeoutError(self._lock_path, self._lock_timeout) from exc
        try:
            yield
        finally:
            self._lock.release()
            if not self._lock.is_locked:
                pending, self._pending_changes = self._pending_changes, []
                for change in pending:
                    self._notify(change)

    def subscribe(
        self, callback: Callable[[DocumentChange], None]
    ) -> Callable[[], None]:
        token = object()
        self._observers[token] = callback

        def unsubscribe() -> None:
            self._observers.pop(token, None)

        return unsubscribe

    def _dispatch_change(self, change: DocumentChange) -> None:
        if self._lock.is_locked:
            self._pending_changes.append(change)
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
