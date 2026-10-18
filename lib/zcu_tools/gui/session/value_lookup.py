"""Read-only value lookup for session-scoped, resolve-once values.

This module is a pure session-layer leaf. It intentionally knows nothing about
Qt, app/measure cfg trees, devices, or predictors; concrete owners register small,
side-effect-free providers through ``ValueRegistry`` and callers receive only the
``ValueLookup`` read interface.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast

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
        return name_from_type(self.type_)


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

    def info(self: _Entry[ScalarValue]) -> ValueInfo:
        return ValueInfo(
            key=self.key.path,
            type_=self.key.type_,
            owner=self.owner,
            description=self.description,
        )


class Registration:
    """Idempotent handle for one provider registration."""

    def __init__(self, release: Callable[[], None]) -> None:
        """Store a zero-argument action that releases one registration.

        ``release`` must return None and perform the registration's cleanup when
        called. Construction stores it without invoking it; errors from the
        action propagate from ``close``.
        """
        self._release = release
        self._closed = False

    def close(self) -> None:
        """Invoke the release action at most once; subsequent calls do nothing.

        Mark the handle closed before invoking the action, including for
        reentrant calls. Action errors propagate and leave the handle closed,
        so a failed release is not retried.
        """
        if self._closed:
            return
        self._closed = True
        self._release()


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
        """Register a lazy scalar provider and return its closeable handle.

        ``key`` identifies the path and supported scalar type. ``provider`` is a
        synchronous, read-only, zero-argument callable returning that type; it
        must not perform hardware I/O. ``owner`` is a non-blank lifecycle group
        name and ``description`` is optional metadata. Registration does not
        evaluate the provider; lookup validates its result and reports failures.

        Closing the returned handle removes only this key's registration, not
        other keys for the owner or a later registration at the same key.

        Raises:
            ValueError: The key path or owner is empty or blank.
            ValueTypeError: The key uses an unsupported scalar type.
            DuplicateValueKey: The key path is already registered.
        """
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
            partial(self._unregister_key, entry.key.path, entry.owner, entry.token)
        )

    def unregister_owner(self, owner: str) -> None:
        _validate_owner(owner)
        for key in tuple(self._owner_keys.pop(owner, set())):
            entry = self._entries.get(key)
            if entry is not None and entry.owner == owner:
                del self._entries[key]

    def replace_owner(
        self, owner: str, providers: Iterable[ValueProviderSpec[Any]]
    ) -> None:
        """Replace all sources for an owner without evaluating providers.

        ``owner`` must be a non-empty, non-blank string. ``providers`` is an
        iterable of specs whose owners all equal ``owner``; specs may use
        different supported scalar types. An empty iterable removes the owner's
        sources.

        Consume and validate the entire group before removing the old sources
        and installing replacements. Iteration and entry-construction errors
        propagate.

        Raises:
            ValueError: The owner is empty or blank, or a spec's owner differs.
            DuplicateValueKey: Keys repeat within the group or conflict with
                sources belonging to another owner.
        """
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
            registered_type = entry.key.type_
            requested_type = _ensure_supported_type(requested.path, requested.type_)
            raise ValueTypeError(
                requested.path,
                f"Value source {requested.path!r} has type "
                f"{name_from_type(registered_type)!r}, requested "
                f"{name_from_type(requested_type)!r}",
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
        # tolerated(cast): Validated scalar keys cross the invariant _Entry[T] registry boundary.
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
            f"Value source {ref.key!r} requested as {name_from_type(ref_type)!r} "
            f"but target expects {name_from_type(target_type)!r}",
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


def _ensure_supported_type(key: str, type_: type[T]) -> ScalarType:
    if (
        type_ is not int
        and type_ is not float
        and type_ is not str
        and type_ is not bool
    ):
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
        # tolerated(cast): Exact bool validation cannot narrow the unconstrained requested T.
        return cast(T, value)
    if type_ is int:
        if type(value) is not int:
            raise ValueTypeError(key, f"Value source {key!r} did not return an int")
        # tolerated(cast): Exact int validation cannot narrow the unconstrained requested T.
        return cast(T, value)
    if type_ is float:
        if type(value) is int or type(value) is float:
            # tolerated(cast): Native float conversion cannot narrow the unconstrained requested T.
            return cast(T, float(value))
        raise ValueTypeError(key, f"Value source {key!r} did not return a float")
    if type_ is str:
        if type(value) is not str:
            raise ValueTypeError(key, f"Value source {key!r} did not return a str")
        # tolerated(cast): Exact str validation cannot narrow the unconstrained requested T.
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


def name_from_type(type_: ScalarType) -> str:
    if type_ is int:
        return "int"
    if type_ is float:
        return "float"
    if type_ is str:
        return "str"
    if type_ is bool:
        return "bool"
    raise AssertionError(f"Unsupported type {type_!r}")
