"""Typed optimistic transactions over an existing round-trip YAML document."""

from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

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
        yaml = YAML(typ="rt")
        with path.open(encoding="utf-8") as stream:
            document = TypeAdapter(YamlMap).validate_python(yaml.load(stream))
        validate_header(
            document,
            expected_format=format,
            supported_version=supported_version,
            source=path,
        )
        self._snapshot = model.model_validate(document)

    def snapshot(self) -> T:
        return self._snapshot.model_copy(deep=True)

    @contextmanager
    def edit(self) -> Generator[T]:
        if self._editing:
            raise RuntimeError(f"{self._path}: nested edits are not allowed")
        self._editing = True
        try:
            with self.locked():
                _, base = self._read()
                draft = base.model_copy(deep=True)
            yield draft
            base_values = TypeAdapter(YamlMap).validate_python(base.model_dump())
            draft_values = TypeAdapter(YamlMap).validate_python(draft.model_dump())
            patches = tuple(_changes(base_values, draft_values))
            with self.locked():
                document, _ = self._read()
                for path, value in patches:
                    _apply(document, path, value)
                snapshot = self._model.model_validate(document)
                if patches:
                    self._write(document)
                self._snapshot = snapshot
        finally:
            self._editing = False

    def _read(self) -> tuple[YamlMap, T]:
        yaml = YAML(typ="rt")
        with self._path.open(encoding="utf-8") as stream:
            document = TypeAdapter(YamlMap).validate_python(yaml.load(stream))
        validate_header(
            document,
            expected_format=self._format,
            supported_version=self._supported_version,
            source=self._path,
        )
        return document, self._model.model_validate(document)

    def _write(self, document: YamlMap) -> None:
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

    def refresh(self) -> bool:
        raise NotImplementedError

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
        raise NotImplementedError
