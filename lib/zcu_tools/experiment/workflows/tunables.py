"""Validated native model snapshots and journal-before-publication replacement."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from threading import RLock

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from .journal import TunablesChanged
from .models import Actor, RevisionConflict, TunablesSnapshot

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
    and append, so model values and revision are published only after their
    journal line.

    This package-internal seam hides whole-model validation,
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
        """Return detached JSON values at one revision."""
        with self._lock:
            return TunablesSnapshot(self._revision, _model_json(self._values))

    def update(
        self,
        tunables: BaseModel,
        *,
        expected_revision: int,
        actor: Actor,
    ) -> TunablesSnapshot:
        """Revalidate a complete declared native model, journal, then replace it.

        tunables must be an instance of model_type. Wrong model types raise
        TypeError; invalid fields or nonfinite values raise ValidationError.
        Values are revalidated from fields, not trusted model instances.
        expected_revision must be a nonnegative integer; stale versions raise
        RevisionConflict. actor attributes the complete replacement.
        append errors propagate unchanged and publish nothing.
        Every success increments revision once, even for identical values.
        """
        with self._lock:
            if type(expected_revision) is not int or expected_revision < 0:
                raise ValueError("expected_revision must be a nonnegative integer")
            if expected_revision != self._revision:
                raise RevisionConflict(expected_revision, self._revision)
            if not isinstance(tunables, self._model_type):
                raise TypeError("Expected an instance of the declared tunables model")
            model = deepcopy(
                self._model_type.model_validate(
                    tunables.model_dump(mode="python"), by_name=True, by_alias=False
                )
            )
            after = _model_json(model)
            event = TunablesChanged(
                actor, self._revision, self._revision + 1, deepcopy(after)
            )
            self._append(event)
            self._values = model
            self._revision += 1
            return TunablesSnapshot(self._revision, after)


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
