"""Typed optimistic transactions over an existing round-trip YAML document."""

import logging
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
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
    elif base != draft:
        yield path, draft


def _lookup(document: YamlMap, path: FieldPath) -> YamlValue | _Missing:
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
            base_values = TypeAdapter(YamlMap).validate_python(base.model_dump())
            draft_values = TypeAdapter(YamlMap).validate_python(draft.model_dump())
            patches = tuple(_changes(base_values, draft_values))
            with self.locked():
                document, _ = self._read()
                for path, _ in patches:
                    original = _lookup(base_document, path)
                    current = _lookup(document, path)
                    if original != current:
                        raise ConflictError(self._path, path, original, current)
                units = self._unit_specs(document)
                for path, value in patches:
                    if path in units:
                        value = self._scale_value(
                            value, _unit_factor(units[path]), path
                        )
                    _apply(document, path, value)
                snapshot = self._model_snapshot(document)
                if patches:
                    self._write(document)
                self._snapshot = snapshot
                self._document = document
        finally:
            self._editing = False
        if patches:
            self._notify(
                DocumentChange(self._path, tuple(path for path, _ in patches), "commit")
            )

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
        return self._model.model_validate(self._working_values(document), extra=extra)

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
            self._notify(DocumentChange(self._path, paths, "refresh"))
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

    def subscribe(
        self, callback: Callable[[DocumentChange], None]
    ) -> Callable[[], None]:
        token = object()
        self._observers[token] = callback

        def unsubscribe() -> None:
            self._observers.pop(token, None)

        return unsubscribe

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
