"""Read-only value lookup for session-scoped, resolve-once values.

This module is a pure session-layer leaf. It intentionally knows nothing about
Qt, app/main cfg trees, devices, or predictors; concrete owners register small,
side-effect-free providers through ``ValueRegistry`` and callers receive only the
``ValueLookup`` read interface.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar, cast

from zcu_tools.gui.expected_error import ExpectedError, ExpectedErrorCategory

ScalarValue: TypeAlias = int | float | str | bool
ScalarType: TypeAlias = type[int] | type[float] | type[str] | type[bool]

T = TypeVar("T")


class _Missing:
    pass


MISSING = _Missing()


class ValueLookupError(RuntimeError):
    """Base class for value lookup failures.

    Only selected concrete leaves opt in to the expected-error taxonomy.
    """

    def __init__(self, key: str, message: str) -> None:
        self.key = key
        super().__init__(message)


class MissingValue(ValueLookupError, ExpectedError):
    """The requested key is not registered."""

    category = ExpectedErrorCategory.INVALID_INPUT


class UnavailableValue(ValueLookupError, ExpectedError):
    """The key is known but cannot be read in the current session state."""

    category = ExpectedErrorCategory.FAILED_PRECONDITION


class ValueTypeError(ValueLookupError, TypeError, ExpectedError):
    """The registered or returned value does not match the requested type."""

    category = ExpectedErrorCategory.INVALID_INPUT


class ProviderError(ValueLookupError):
    """A provider raised an unexpected error while computing a value."""

    def __init__(self, key: str, owner: str, cause: BaseException) -> None:
        self.owner = owner
        self.cause = cause
        super().__init__(
            key,
            f"Provider for value source {key!r} owned by {owner!r} failed: {cause}",
        )


class DuplicateValueKey(ValueLookupError):
    """A registration attempted to overwrite an existing key."""


@dataclass(frozen=True)
class ValueKey(Generic[T]):
    path: str
    type_: type[T]

    def __post_init__(self) -> None:
        _validate_path(self.path)
        _ensure_supported_type(self.path, self.type_)


@dataclass(frozen=True)
class ValueRef:
    """Resolve-once reference to a registered value source."""

    key: str
    type_name: str | None = None

    def __post_init__(self) -> None:
        _validate_path(self.key)
        if self.type_name is not None:
            _type_from_name(self.type_name, key=self.key)


@dataclass(frozen=True)
class ValueInfo:
    key: str
    type_: ScalarType
    owner: str
    description: str = ""

    @property
    def type_name(self) -> str:
        return _name_from_type(self.type_)


@dataclass(frozen=True)
class ValueProviderSpec(Generic[T]):
    key: ValueKey[T]
    provider: Callable[[], T]
    owner: str
    description: str = ""

    def __post_init__(self) -> None:
        _validate_owner(self.owner)


class ValueLookup(Protocol):
    """Read-only surface exposed to adapters, GUI, and remote handlers."""

    def get(self, key: ValueKey[T], *, default: T | _Missing = MISSING) -> T: ...

    def get_as(
        self, path: str, type_: type[T], *, default: T | _Missing = MISSING
    ) -> T: ...

    def describe(self) -> tuple[ValueInfo, ...]: ...


@dataclass(frozen=True)
class _Entry(Generic[T]):
    key: ValueKey[T]
    provider: Callable[[], T]
    owner: str
    description: str
    token: object

    def info(self) -> ValueInfo:
        return ValueInfo(
            key=self.key.path,
            type_=cast(ScalarType, self.key.type_),
            owner=self.owner,
            description=self.description,
        )


class Registration:
    """Idempotent handle for one provider registration."""

    def __init__(
        self,
        registry: ValueRegistry,
        *,
        key: str,
        owner: str,
        token: object,
    ) -> None:
        self._registry = registry
        self._key = key
        self._owner = owner
        self._token = token
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._registry._unregister_key(self._key, self._owner, self._token)


class EmptyValueLookup:
    """Null lookup installed before a real registry is wired."""

    def get(self, key: ValueKey[T], *, default: T | _Missing = MISSING) -> T:
        if not isinstance(default, _Missing):
            return default
        raise MissingValue(key.path, f"Value source {key.path!r} is not registered")

    def get_as(
        self, path: str, type_: type[T], *, default: T | _Missing = MISSING
    ) -> T:
        return self.get(ValueKey(path, type_), default=default)

    def describe(self) -> tuple[ValueInfo, ...]:
        return ()


class ValueRegistry(ValueLookup):
    """Mutable owner-scoped registry behind the read-only ``ValueLookup`` seam."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry[ScalarValue]] = {}
        self._owner_keys: dict[str, set[str]] = {}

    def register(
        self,
        key: ValueKey[T],
        provider: Callable[[], T],
        *,
        owner: str,
        description: str = "",
    ) -> Registration:
        spec = ValueProviderSpec(
            key=key,
            provider=provider,
            owner=owner,
            description=description,
        )
        entry = self._entry_from_spec(spec)
        if entry.key.path in self._entries:
            raise DuplicateValueKey(
                entry.key.path, f"Value source {entry.key.path!r} is already registered"
            )
        self._install_entry(entry)
        return Registration(
            self, key=entry.key.path, owner=entry.owner, token=entry.token
        )

    def unregister_owner(self, owner: str) -> None:
        _validate_owner(owner)
        for key in tuple(self._owner_keys.pop(owner, set())):
            entry = self._entries.get(key)
            if entry is not None and entry.owner == owner:
                del self._entries[key]

    def replace_owner(self, owner: str, providers: Iterable[ValueProviderSpec]) -> None:
        _validate_owner(owner)
        entries = tuple(self._entry_from_spec(spec) for spec in providers)
        new_keys: set[str] = set()
        for entry in entries:
            if entry.owner != owner:
                raise ValueError(
                    f"Provider for key {entry.key.path!r} belongs to owner "
                    f"{entry.owner!r}, expected {owner!r}"
                )
            if entry.key.path in new_keys:
                raise DuplicateValueKey(
                    entry.key.path,
                    f"Owner {owner!r} registered duplicate value source "
                    f"{entry.key.path!r}",
                )
            existing = self._entries.get(entry.key.path)
            if existing is not None and existing.owner != owner:
                raise DuplicateValueKey(
                    entry.key.path,
                    f"Value source {entry.key.path!r} is already owned by "
                    f"{existing.owner!r}",
                )
            new_keys.add(entry.key.path)

        # Atomic with respect to registry invariants: validate every new provider
        # before removing the old owner group or installing replacements.
        self.unregister_owner(owner)
        for entry in entries:
            self._install_entry(entry)

    def get(self, key: ValueKey[T], *, default: T | _Missing = MISSING) -> T:
        try:
            return self._get_required(key)
        except (MissingValue, UnavailableValue):
            if not isinstance(default, _Missing):
                return default
            raise

    def get_as(
        self, path: str, type_: type[T], *, default: T | _Missing = MISSING
    ) -> T:
        return self.get(ValueKey(path, type_), default=default)

    def describe(self) -> tuple[ValueInfo, ...]:
        return tuple(self._entries[key].info() for key in sorted(self._entries))

    def _get_required(self, requested: ValueKey[T]) -> T:
        entry = self._entries.get(requested.path)
        if entry is None:
            raise MissingValue(
                requested.path, f"Value source {requested.path!r} is not registered"
            )
        if entry.key.type_ is not requested.type_:
            registered_type = cast(ScalarType, entry.key.type_)
            requested_type = cast(
                ScalarType,
                _ensure_supported_type(requested.path, requested.type_),
            )
            raise ValueTypeError(
                requested.path,
                f"Value source {requested.path!r} has type "
                f"{_name_from_type(registered_type)!r}, requested "
                f"{_name_from_type(requested_type)!r}",
            )
        try:
            value = entry.provider()
        except UnavailableValue:
            raise
        except ValueLookupError:
            raise
        except Exception as exc:
            raise ProviderError(entry.key.path, entry.owner, exc) from exc
        return _coerce_value(entry.key.path, value, requested.type_)

    def _entry_from_spec(self, spec: ValueProviderSpec[T]) -> _Entry[ScalarValue]:
        return cast(
            _Entry[ScalarValue],
            _Entry(
                key=ValueKey(spec.key.path, spec.key.type_),
                provider=spec.provider,
                owner=spec.owner,
                description=spec.description,
                token=object(),
            ),
        )

    def _install_entry(self, entry: _Entry[ScalarValue]) -> None:
        self._entries[entry.key.path] = entry
        self._owner_keys.setdefault(entry.owner, set()).add(entry.key.path)

    def _unregister_key(self, key: str, owner: str, token: object) -> None:
        entry = self._entries.get(key)
        if entry is None or entry.owner != owner or entry.token is not token:
            return
        del self._entries[key]
        owner_keys = self._owner_keys.get(owner)
        if owner_keys is not None:
            owner_keys.discard(key)
            if not owner_keys:
                del self._owner_keys[owner]


def decode_value_ref(raw: object) -> ValueRef | None:
    """Decode an explicit wire/internal value-ref tag.

    Returns ``None`` for non-value-ref objects so callers can compose this with
    existing decoders. Malformed value-ref tags raise ``ValueError``.
    """

    if not isinstance(raw, Mapping):
        return None
    if raw.get("__kind") != "value_ref":
        return None
    key = raw.get("key")
    if not isinstance(key, str):
        raise ValueError("value_ref requires a string 'key'")
    type_name = raw.get("type", raw.get("type_name"))
    if type_name is not None and not isinstance(type_name, str):
        raise ValueError("value_ref 'type' must be a string when provided")
    return ValueRef(key=key, type_name=type_name)


def parse_value_ref_text(text: str) -> ValueRef | None:
    """Parse the explicit GUI convenience form ``@{source.key}``.

    Parsing is intentionally exact after trimming surrounding whitespace; arbitrary
    strings are not tokenized.
    """

    stripped = text.strip()
    if not (stripped.startswith("@{") and stripped.endswith("}")):
        return None
    key = stripped[2:-1].strip()
    if not key:
        raise ValueError("value_ref text requires a non-empty key")
    return ValueRef(key=key)


def resolve_value_ref(
    ref: ValueRef,
    lookup: ValueLookup,
    *,
    target_type: ScalarType | None = None,
) -> ScalarValue:
    """Resolve a ``ValueRef`` immediately to a concrete scalar."""

    ref_type = _type_from_name(ref.type_name, key=ref.key) if ref.type_name else None
    if ref_type is not None and target_type is not None and ref_type is not target_type:
        raise ValueTypeError(
            ref.key,
            f"Value source {ref.key!r} requested as {_name_from_type(ref_type)!r} "
            f"but target expects {_name_from_type(target_type)!r}",
        )
    resolved_type = target_type or ref_type or _registered_type(lookup, ref.key)
    return lookup.get_as(ref.key, resolved_type)


def _registered_type(lookup: ValueLookup, key: str) -> ScalarType:
    for info in lookup.describe():
        if info.key == key:
            return info.type_
    raise MissingValue(key, f"Value source {key!r} is not registered")


def _validate_path(path: str) -> None:
    if not path or not path.strip():
        raise ValueError("Value source key must be a non-empty string")


def _validate_owner(owner: str) -> None:
    if not owner or not owner.strip():
        raise ValueError("Value source owner must be a non-empty string")


def _ensure_supported_type(key: str, type_: type[T]) -> type[T]:
    if type_ not in (int, float, str, bool):
        raise ValueTypeError(
            key,
            f"Value source {key!r} uses unsupported type {type_.__name__!r}; "
            "only int, float, str, and bool are supported",
        )
    return type_


def _coerce_value(key: str, value: object, type_: type[T]) -> T:
    _ensure_supported_type(key, type_)
    if type_ is bool:
        if type(value) is not bool:
            raise ValueTypeError(key, f"Value source {key!r} did not return a bool")
        return cast(T, value)
    if type_ is int:
        if type(value) is not int:
            raise ValueTypeError(key, f"Value source {key!r} did not return an int")
        return cast(T, value)
    if type_ is float:
        if type(value) in (int, float):
            return cast(T, float(cast(int | float, value)))
        raise ValueTypeError(key, f"Value source {key!r} did not return a float")
    if type_ is str:
        if type(value) is not str:
            raise ValueTypeError(key, f"Value source {key!r} did not return a str")
        return cast(T, value)
    raise AssertionError(f"Unsupported type {type_!r}")


def _type_from_name(type_name: str, *, key: str) -> ScalarType:
    mapping: dict[str, ScalarType] = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }
    try:
        return mapping[type_name]
    except KeyError as exc:
        raise ValueTypeError(
            key,
            f"Value source {key!r} uses unknown type name {type_name!r}",
        ) from exc


def _name_from_type(type_: ScalarType) -> str:
    if type_ is int:
        return "int"
    if type_ is float:
        return "float"
    if type_ is str:
        return "str"
    if type_ is bool:
        return "bool"
    raise AssertionError(f"Unsupported type {type_!r}")
