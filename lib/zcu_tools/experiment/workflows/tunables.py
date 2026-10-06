"""Validated tunables snapshots and journal-before-publication batch updates."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from threading import RLock

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from .declaration import validate_tunable_path
from .journal import ChangedValue, TunablesChanged
from .models import Actor, RevisionConflict, TunableChange, TunablesSnapshot

_json_adapter = TypeAdapter(JsonValue)


@dataclass(frozen=True)
class CapturedTunables[T: BaseModel]:
    """One invocation's isolated input, unaffected by later edits.

    revision is the captured version, starting at zero. model is a deepcopy of
    the validated complete tunables model. Only the execution thread uses it.
    """

    revision: int
    model: T


class TunableValues[T: BaseModel]:
    """Own a complete tunables model and its monotonically increasing revision.

    model_type is the workflow-declared pydantic type. values is revalidated
    from its fields and detached at construction, creating revision zero.
    control_lock is the Engine's shared RLock. append must append/flush one
    TunablesChanged event or raise. The same lock serializes capture, update,
    and the append, so the model is published only after its journal line.

    This package-internal seam hides schema paths, whole-model validation,
    detached JSON projections, and atomic revision publication. Engine retains
    lifecycle/run-id policy and turns append errors into run failure.
    """

    def __init__(
        self,
        model_type: type[T],
        values: T,
        control_lock: RLock,
        append: Callable[[TunablesChanged], None],
    ) -> None:
        self._model_type = model_type
        self._values = deepcopy(
            model_type.model_validate(
                values.model_dump(mode="python"), by_name=True, by_alias=False
            )
        )
        _model_json(self._values)
        self._revision = 0
        self._lock = control_lock
        self._append = append

    @property
    def revision(self) -> int:
        """Return the current version without re-encoding the model."""
        with self._lock:
            return self._revision

    def capture(self) -> CapturedTunables[T]:
        """Capture the revision and deepcopy model for one invocation.

        Later updates do not mutate this copy. Deepcopy errors propagate to the
        Engine as contract violations, never as record-encoding degradation.
        """
        with self._lock:
            return CapturedTunables(self._revision, deepcopy(self._values))

    def snapshot(self) -> TunablesSnapshot:
        """Return the current revision and detached complete JSON model values."""
        with self._lock:
            return TunablesSnapshot(self._revision, _model_json(self._values))

    def update(
        self,
        changes: tuple[TunableChange, ...],
        *,
        expected_revision: int,
        actor: Actor,
    ) -> TunablesSnapshot:
        """Validate and journal one nonempty leaf-replacement batch, then publish.

        paths use model field names and dot components, not aliases or array
        indices. Lists/tuples are replaced as a whole. Unknown paths, model-layer
        replacements, overlap, duplicates, and empty batches raise ValueError.
        Updating through an absent optional model also raises ValueError.
        None is a value, not deletion; model validation decides if it is legal.

        expected_revision is a nonnegative integer. A stale value raises
        RevisionConflict with expected/actual; schema and nonfinite errors
        retain pydantic ValidationError locations. actor attributes the batch.
        The complete candidate is validated, including cross-field rules.
        append errors propagate with their original cause and publish nothing.
        A successful batch increments revision once, including same-value edits.
        """
        with self._lock:
            if type(expected_revision) is not int or expected_revision < 0:
                raise ValueError("expected_revision must be a nonnegative integer")
            if expected_revision != self._revision:
                raise RevisionConflict(expected_revision, self._revision)
            _check_batch(changes)
            before = _model_json(self._values)
            candidate = deepcopy(before)
            for change in changes:
                validate_tunable_path(self._model_type, change.path)
                _replace_leaf(candidate, change)
            model = self._model_type.model_validate(
                candidate, by_name=True, by_alias=False
            )
            after = _model_json(model)
            applied = tuple(
                ChangedValue(
                    change.path,
                    deepcopy(_leaf_value(before, change.path)),
                    deepcopy(_leaf_value(after, change.path)),
                )
                for change in changes
            )
            event = TunablesChanged(actor, self._revision, self._revision + 1, applied)
            self._append(event)
            self._values = model
            self._revision += 1
            return TunablesSnapshot(self._revision, after)


def _check_batch(changes: tuple[TunableChange, ...]) -> None:
    if not changes:
        raise ValueError("Tunable updates require a nonempty batch")
    paths = tuple(change.path for change in changes)
    unique_paths = set(paths)
    if len(unique_paths) != len(paths):
        raise ValueError("Duplicate tunable paths in batch")
    for path in paths:
        parts = path.split(".")
        for end in range(1, len(parts)):
            if ".".join(parts[:end]) in unique_paths:
                raise ValueError("Overlapping parent/child tunable paths")


def _model_json(model: BaseModel) -> dict[str, JsonValue]:
    values = _json_adapter.validate_python(
        model.model_dump(mode="json", by_alias=False, warnings="error")
    )
    if not isinstance(values, dict):
        raise ValueError("Tunables must project to a JSON model object")
    _check_finite(values, (), type(model).__name__)
    return values


def _check_finite(
    value: JsonValue, location: tuple[str | int, ...], title: str
) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise ValidationError.from_exception_data(
            title, [{"type": "finite_number", "loc": location, "input": value}]
        )
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite(child, (*location, key), title)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite(child, (*location, index), title)


def _leaf_parent(values: dict[str, JsonValue], path: str) -> dict[str, JsonValue]:
    current = values
    for part in path.split(".")[:-1]:
        child = current[part]
        if not isinstance(child, dict):
            raise ValueError(f"Cannot update through an absent model: {path!r}")
        current = child
    return current


def _leaf_value(values: dict[str, JsonValue], path: str) -> JsonValue:
    return _leaf_parent(values, path)[path.split(".")[-1]]


def _replace_leaf(values: dict[str, JsonValue], change: TunableChange) -> None:
    parent = _leaf_parent(values, change.path)
    name = change.path.split(".")[-1]
    parent[name] = deepcopy(change.value)
