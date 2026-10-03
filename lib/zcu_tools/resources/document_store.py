"""Typed optimistic transactions over an existing round-trip YAML document."""

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, TypeAdapter
from ruamel.yaml import YAML

from zcu_tools.format_version import FormatVersion, YamlMap, YamlValue, validate_header

type FieldPath = tuple[str, ...]


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


class DocumentStore[T: BaseModel]:
    def __init__(
        self,
        path: Path,
        model: type[T],
        *,
        format: str,
        supported_version: FormatVersion = FormatVersion(1, 0),
        units: Mapping[FieldPath, UnitSpec] | UnitResolver | None = None,
        validate: Callable[[T], None] | None = None,
        lock_path: Path | None = None,
        lock_timeout: float = 10.0,
    ) -> None:
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

    def edit(self) -> AbstractContextManager[T]:
        raise NotImplementedError

    def refresh(self) -> bool:
        raise NotImplementedError

    def locked(self) -> AbstractContextManager[None]:
        raise NotImplementedError

    def subscribe(
        self, callback: Callable[[DocumentChange], None]
    ) -> Callable[[], None]:
        raise NotImplementedError
