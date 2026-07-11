"""Canonical settable-path vocabulary for a live :class:`CfgDraft`."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

from ..model import CfgSectionSpec, DirectValue, EvalValue, ReferenceSpec
from ..reference_key import make_custom_reference_key, parse_custom_reference_key
from .fields import (
    CenteredSweepField,
    CfgField,
    LiteralField,
    ScalarField,
    SectionField,
    SweepField,
)
from .range import CenteredSweepEditor, SweepEditor
from .reference import ReferenceField

_SWEEP_EDGES = ("start", "stop", "expts", "step")
_CENTERED_SWEEP_EDGES = ("center", "span", "expts", "step")
_MAX_SUGGESTIONS = 3


class SettablePathError(InvalidInputError):
    """A canonical cfg path or its input value is invalid."""

    def __init__(self, message: str, *, reason_code: str = "invalid_settable_path"):
        super().__init__(message, reason_code=reason_code)


class LegacySettablePathError(SettablePathError):
    """A removed path spelling was recognized without being executed."""

    def __init__(self, path: str, replacement: str) -> None:
        self.path = path
        self.replacement = replacement
        super().__init__(
            f"path {path!r} uses a removed legacy segment; use {replacement!r}",
            reason_code="legacy_settable_path",
        )


class SettableTargetUnavailable(FailedPreconditionError):
    """The current reference shape does not expose the requested target."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="settable_target_unavailable")


class SettableTargetKind(StrEnum):
    SCALAR = "scalar"
    SWEEP_EDGE = "sweep_edge"
    REFERENCE_KEY = "reference_key"


@dataclass(frozen=True, slots=True)
class SettableTarget:
    """One nominal, live mutation target in the canonical dotted-path grammar."""

    path: str
    kind: SettableTargetKind
    value_type: type
    affects_path_shape: bool
    _get: Callable[[], object] = field(repr=False, compare=False)
    _set: Callable[[object], None] = field(repr=False, compare=False)
    _choices: Callable[[], tuple[object, ...] | None] = field(repr=False, compare=False)

    def get_value(self) -> object:
        return self._get()

    def choices(self) -> tuple[object, ...] | None:
        return self._choices()

    def set_value(self, value: object) -> None:
        self._set(value)


@dataclass(frozen=True, slots=True)
class _TargetIndex:
    targets: tuple[SettableTarget, ...]
    container_paths: frozenset[str]
    unavailable_references: frozenset[str]


def iter_settable_targets(root: SectionField) -> Iterator[SettableTarget]:
    """Iterate canonical targets in stable field/edge insertion order."""
    yield from _build_target_index(root).targets


def resolve_settable_target(root: SectionField, path: str) -> SettableTarget:
    """Resolve exactly one canonical path using the same traversal as listing."""
    if not isinstance(path, str) or not path:
        raise SettablePathError("settable path must be a non-empty string")
    if any(not segment for segment in path.split(".")):
        raise SettablePathError(f"settable path {path!r} contains an empty segment")

    index = _build_target_index(root)
    by_path = {target.path: target for target in index.targets}
    target = by_path.get(path)
    if target is not None:
        return target

    replacement = _legacy_replacement(path, by_path)
    if replacement is not None:
        raise LegacySettablePathError(path, replacement)

    for ref_path in index.unavailable_references:
        if path.startswith(ref_path + ".") and path != ref_path + ".ref":
            raise SettableTargetUnavailable(
                f"reference at {ref_path!r} has no editable sub-fields for its "
                "current key"
            )

    if path in index.container_paths:
        descendants = [p for p in by_path if p.startswith(path + ".")]
        hint = ", ".join(repr(p) for p in descendants[:3])
        suffix = f"; descend to {hint}" if hint else ""
        raise SettablePathError(f"path {path!r} is not a settable leaf{suffix}")

    message = f"unknown settable path {path!r}"
    suggestions = _suggest_paths(path, tuple(by_path))
    if suggestions:
        message += (
            "; did you mean " + ", ".join(repr(item) for item in suggestions) + "?"
        )
    raise SettablePathError(message)


def _build_target_index(root: SectionField) -> _TargetIndex:
    _validate_spec_keys(root.spec)
    targets: list[SettableTarget] = []
    containers: set[str] = set()
    unavailable: set[str] = set()
    _collect_targets(root, "", targets, containers, unavailable)
    return _TargetIndex(tuple(targets), frozenset(containers), frozenset(unavailable))


def _collect_targets(
    field: CfgField,
    path: str,
    targets: list[SettableTarget],
    containers: set[str],
    unavailable: set[str],
) -> None:
    if isinstance(field, LiteralField):
        if path:
            containers.add(path)
        return
    if isinstance(field, ScalarField):
        targets.append(_scalar_target(path, field))
        return
    if isinstance(field, SweepField):
        containers.add(path)
        targets.extend(_sweep_targets(path, field))
        return
    if isinstance(field, CenteredSweepField):
        containers.add(path)
        targets.extend(_centered_sweep_targets(path, field))
        return
    if isinstance(field, ReferenceField):
        containers.add(path)
        targets.append(_reference_target(path, field))
        if field.sub_field is None:
            unavailable.add(path)
            return
        for key, child in field.sub_field.fields.items():
            _collect_targets(child, _join(path, key), targets, containers, unavailable)
        return
    if isinstance(field, SectionField):
        if path:
            containers.add(path)
        for key, child in field.fields.items():
            _collect_targets(child, _join(path, key), targets, containers, unavailable)


def _scalar_target(path: str, scalar: ScalarField) -> SettableTarget:
    def set_value(value: object) -> None:
        if not isinstance(value, (DirectValue, EvalValue)):
            value = DirectValue(value)
        try:
            scalar.set_value(value)
        except (TypeError, ValueError) as exc:
            raise SettablePathError(f"invalid value for {path!r}: {exc}") from exc

    return SettableTarget(
        path=path,
        kind=SettableTargetKind.SCALAR,
        value_type=scalar.spec.type,
        affects_path_shape=False,
        _get=scalar.get_value,
        _set=set_value,
        _choices=scalar.available_options,
    )


def _sweep_targets(path: str, sweep: SweepField) -> list[SettableTarget]:
    def get(edge: str) -> object:
        return getattr(sweep.get_value(), edge)

    def set_edge(edge: str, value: object) -> None:
        current = sweep.get_value()
        try:
            if edge == "expts":
                if type(value) is not int:
                    raise TypeError("expts must be an integer")
                updated = SweepEditor.update_expts(current, value)
            elif edge == "step":
                updated = SweepEditor.update_step(current, _number(value, edge))
            elif edge == "start":
                updated = SweepEditor.update_start(current, _edge_value(value, edge))
            else:
                updated = SweepEditor.update_stop(current, _edge_value(value, edge))
            sweep.set_value(updated)
        except (TypeError, ValueError) as exc:
            raise SettablePathError(
                f"invalid value for {_join(path, edge)!r}: {exc}"
            ) from exc

    return [
        SettableTarget(
            path=_join(path, edge),
            kind=SettableTargetKind.SWEEP_EDGE,
            value_type=int if edge == "expts" else float,
            affects_path_shape=False,
            _get=lambda edge=edge: get(edge),
            _set=lambda value, edge=edge: set_edge(edge, value),
            _choices=lambda: None,
        )
        for edge in _SWEEP_EDGES
    ]


def _centered_sweep_targets(
    path: str, sweep: CenteredSweepField
) -> list[SettableTarget]:
    def get(edge: str) -> object:
        return getattr(sweep.get_value(), edge)

    def set_edge(edge: str, value: object) -> None:
        current = sweep.get_value()
        try:
            if edge == "expts":
                if type(value) is not int:
                    raise TypeError("expts must be an integer")
                updated = CenteredSweepEditor.update_expts(current, value)
            elif edge == "step":
                updated = CenteredSweepEditor.update_step(current, _number(value, edge))
            elif edge == "center":
                updated = CenteredSweepEditor.update_center(
                    current, _edge_value(value, edge)
                )
            else:
                updated = CenteredSweepEditor.update_span(current, _number(value, edge))
            sweep.set_value(updated)
        except (TypeError, ValueError) as exc:
            raise SettablePathError(
                f"invalid value for {_join(path, edge)!r}: {exc}"
            ) from exc

    return [
        SettableTarget(
            path=_join(path, edge),
            kind=SettableTargetKind.SWEEP_EDGE,
            value_type=int if edge == "expts" else float,
            affects_path_shape=False,
            _get=lambda edge=edge: get(edge),
            _set=lambda value, edge=edge: set_edge(edge, value),
            _choices=lambda: None,
        )
        for edge in _CENTERED_SWEEP_EDGES
    ]


def _reference_target(path: str, reference: ReferenceField) -> SettableTarget:
    def choices() -> tuple[object, ...]:
        return (
            *(spec.label for spec in reference.spec.allowed),
            *reference.available_keys(),
        )

    def set_value(value: object) -> None:
        if not isinstance(value, str):
            raise SettablePathError(
                f"reference key at {_join(path, 'ref')!r} expects a string"
            )
        try:
            label = parse_custom_reference_key(value)
        except ValueError as exc:
            raise SettablePathError(str(exc)) from exc
        allowed_labels = {spec.label for spec in reference.spec.allowed}
        if label is not None:
            if label not in allowed_labels:
                raise SettablePathError(f"unknown custom reference label: {label!r}")
            key = value
        elif value in allowed_labels:
            key = make_custom_reference_key(value)
        else:
            key = value
        reference.set_chosen_key(key)

    return SettableTarget(
        path=_join(path, "ref"),
        kind=SettableTargetKind.REFERENCE_KEY,
        value_type=str,
        affects_path_shape=True,
        _get=reference.get_chosen_key,
        _set=set_value,
        _choices=choices,
    )


def _number(value: object, edge: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{edge} must be a number")
    return float(value)


def _edge_value(value: object, edge: str) -> float | EvalValue:
    if isinstance(value, EvalValue):
        return value
    return _number(value, edge)


def _legacy_replacement(path: str, targets: dict[str, SettableTarget]) -> str | None:
    parts = path.split(".")
    for legacy in ("sweep", "value"):
        for index, part in enumerate(parts):
            if part != legacy:
                continue
            candidate = ".".join((*parts[:index], *parts[index + 1 :]))
            if candidate in targets:
                return candidate
    return None


def _suggest_paths(path: str, targets: tuple[str, ...]) -> tuple[str, ...]:
    parts = path.split(".")
    leaf = parts[-1]
    existing_prefix: list[str] = []
    for part in parts[:-1]:
        candidate = ".".join((*existing_prefix, part))
        if any(p == candidate or p.startswith(candidate + ".") for p in targets):
            existing_prefix.append(part)
        else:
            break
    prefix = ".".join(existing_prefix)
    matches = sorted(
        {
            target
            for target in targets
            if target.rsplit(".", 1)[-1] == leaf
            and (not prefix or target.startswith(prefix + "."))
        }
    )
    if len(matches) > _MAX_SUGGESTIONS:
        return ()
    return tuple(matches)


def _validate_spec_keys(
    spec: CfgSectionSpec, *, inside_reference: bool = False
) -> None:
    for key, child in spec.fields.items():
        if not key or "." in key or key.startswith("$"):
            raise SettablePathError(
                f"cfg field key {key!r} cannot be represented by canonical paths"
            )
        if inside_reference and key in {"ref", "value"}:
            raise SettablePathError(
                f"reference child key {key!r} collides with canonical path grammar"
            )
        if isinstance(child, ReferenceSpec):
            for allowed in child.allowed:
                _validate_spec_keys(allowed, inside_reference=True)
        elif isinstance(child, CfgSectionSpec):
            _validate_spec_keys(child, inside_reference=inside_reference)


def _join(path: str, segment: str) -> str:
    return f"{path}.{segment}" if path else segment


__all__ = [
    "LegacySettablePathError",
    "SettablePathError",
    "SettableTarget",
    "SettableTargetKind",
    "SettableTargetUnavailable",
]
