"""Validated model updates through the Engine's package-internal tunables seam."""

from __future__ import annotations

from threading import RLock

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)
from zcu_tools.experiment.workflows.journal import TunablesChanged
from zcu_tools.experiment.workflows.models import (
    Actor,
    RevisionConflict,
    TunableChange,
    TunablesSnapshot,
)
from zcu_tools.experiment.workflows.tunables import TunableValues


class Gate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lower: float = 2
    upper: float = 5

    @model_validator(mode="after")
    def ordered(self) -> Gate:
        if self.lower > self.upper:
            raise ValueError("lower must not exceed upper")
        return self


class Tunables(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reps: int = Field(default=2, ge=1)
    gate: Gate = Field(default_factory=Gate)
    points: tuple[float, ...] = (1, 2)
    note: str | None = None


class StrictTunables(Tunables):
    model_config = ConfigDict(extra="forbid", strict=True)


@pytest.fixture
def journal() -> list[TunablesChanged]:
    return []


@pytest.fixture
def values(journal: list[TunablesChanged]) -> TunableValues[Tunables]:
    return TunableValues(Tunables, Tunables(), RLock(), journal.append)


def test_start_revalidates_constructed_instances(
    journal: list[TunablesChanged],
) -> None:
    invalid = Tunables.model_construct(reps=0)
    with pytest.raises(ValidationError) as caught:
        TunableValues(Tunables, invalid, RLock(), journal.append)
    assert caught.value.errors()[0]["loc"] == ("reps",)
    assert journal == []


def test_batch_validates_complete_candidate_and_increments_once(
    values: TunableValues[Tunables], journal: list[TunablesChanged]
) -> None:
    snapshot = values.update(
        (TunableChange("gate.lower", 6), TunableChange("gate.upper", 7)),
        expected_revision=0,
        actor=Actor("agent", "calibrator"),
    )
    assert snapshot.revision == 1
    assert snapshot.values == {
        "reps": 2,
        "gate": {"lower": 6.0, "upper": 7.0},
        "points": [1.0, 2.0],
        "note": None,
    }
    assert len(journal) == 1
    event = journal[0]
    assert event.actor == Actor("agent", "calibrator")
    assert (event.revision_before, event.revision_after) == (0, 1)
    assert [(change.path, change.old, change.new) for change in event.changes] == [
        ("gate.lower", 2.0, 6.0),
        ("gate.upper", 5.0, 7.0),
    ]
    assert values.capture().model.gate == Gate(lower=6, upper=7)


def test_current_invocation_capture_is_not_changed_by_later_update(
    values: TunableValues[Tunables],
) -> None:
    old = values.capture()
    values.update(
        (TunableChange("reps", 7),), expected_revision=0, actor=Actor("user", "owner")
    )
    assert old.revision == 0
    assert old.model.reps == 2
    assert values.capture().revision == 1
    assert values.capture().model.reps == 7
    old.model.gate.lower = -20
    assert values.capture().model.gate.lower == 2


def test_revision_is_published_only_after_append_returns() -> None:
    seen: list[TunablesSnapshot] = []
    events: list[TunablesChanged] = []
    values: TunableValues[Tunables]

    def append(event: TunablesChanged) -> None:
        seen.append(values.snapshot())
        events.append(event)

    values = TunableValues(Tunables, Tunables(), RLock(), append)
    after = values.update(
        (TunableChange("reps", 4),), expected_revision=0, actor=Actor("user", "owner")
    )
    assert len(events) == 1
    assert seen[0].revision == 0
    assert isinstance(seen[0].values, dict)
    assert seen[0].values["reps"] == 2
    assert after.revision == 1
    assert isinstance(after.values, dict)
    assert after.values["reps"] == 4


def test_append_failure_retains_previous_model_revision_and_cause() -> None:
    cause = OSError("journal flush failed")
    events: list[TunablesChanged] = []

    def append(event: TunablesChanged) -> None:
        events.append(event)
        raise cause

    values = TunableValues(Tunables, Tunables(), RLock(), append)
    before = values.snapshot()
    with pytest.raises(OSError, match="journal flush failed") as caught:
        values.update(
            (TunableChange("reps", 4),),
            expected_revision=0,
            actor=Actor("user", "owner"),
        )
    assert caught.value is cause
    assert values.snapshot() == before
    assert len(events) == 1


def test_stale_revision_exposes_expected_and_actual_without_append(
    values: TunableValues[Tunables], journal: list[TunablesChanged]
) -> None:
    values.update(
        (TunableChange("reps", 3),), expected_revision=0, actor=Actor("user", "owner")
    )
    before = values.snapshot()
    with pytest.raises(RevisionConflict) as caught:
        values.update(
            (TunableChange("reps", 4),),
            expected_revision=0,
            actor=Actor("user", "owner"),
        )
    assert caught.value.expected == 0
    assert caught.value.actual == 1
    assert values.snapshot() == before
    assert len(journal) == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ((), "nonempty batch"),
        ((TunableChange("reps", 3), TunableChange("reps", 4)), "Duplicate"),
        ((TunableChange("gate", None), TunableChange("gate.lower", 1)), "Overlapping"),
        ((TunableChange("gate", None),), "entire model"),
        ((TunableChange("unknown", 4),), "Unknown"),
        ((TunableChange("points.0", 4),), "Unknown"),
        ((TunableChange("reps.unknown", 4),), "Unknown"),
        ((TunableChange("", 4),), "nonempty model field"),
        ((TunableChange("gate..lower", 4),), "nonempty model field"),
        ((TunableChange("gate.unknown", 4),), "Unknown"),
    ],
)
def test_illegal_paths_do_not_publish_partial_changes(
    values: TunableValues[Tunables],
    journal: list[TunablesChanged],
    changes: tuple[TunableChange, ...],
    message: str,
) -> None:
    before = values.snapshot()
    with pytest.raises(ValueError, match=message):
        values.update(changes, expected_revision=0, actor=Actor("user", "owner"))
    assert values.snapshot() == before
    assert journal == []


def test_schema_failure_rolls_back_every_field_and_keeps_location(
    values: TunableValues[Tunables], journal: list[TunablesChanged]
) -> None:
    before = values.snapshot()
    with pytest.raises(ValidationError) as caught:
        values.update(
            (TunableChange("note", "new"), TunableChange("reps", 0)),
            expected_revision=0,
            actor=Actor("user", "owner"),
        )
    assert caught.value.errors()[0]["loc"] == ("reps",)
    assert values.snapshot() == before
    assert journal == []


def test_cross_field_validation_failure_does_not_append(
    values: TunableValues[Tunables], journal: list[TunablesChanged]
) -> None:
    with pytest.raises(ValidationError, match="lower must not exceed upper") as caught:
        values.update(
            (TunableChange("gate.lower", 6),),
            expected_revision=0,
            actor=Actor("user", "owner"),
        )
    assert caught.value.errors()[0]["loc"] == ("gate",)
    assert values.capture().model.gate.lower == 2
    assert journal == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_array_leaf_is_rejected_with_model_location(
    values: TunableValues[Tunables], journal: list[TunablesChanged], bad: float
) -> None:
    with pytest.raises(ValidationError) as caught:
        values.update(
            (TunableChange("points", [1.0, bad]),),
            expected_revision=0,
            actor=Actor("user", "owner"),
        )
    assert caught.value.errors()[0]["loc"] == ("points", 1)
    assert caught.value.errors()[0]["type"] == "finite_number"
    assert values.capture().revision == 0
    assert journal == []


def test_start_nonfinite_values_are_not_coerced_to_missing(
    journal: list[TunablesChanged],
) -> None:
    invalid = Tunables(points=(float("inf"),))
    with pytest.raises(ValidationError) as caught:
        TunableValues(Tunables, invalid, RLock(), journal.append)
    assert caught.value.errors()[0]["loc"] == ("points", 0)
    assert journal == []


def test_strict_tuple_model_allows_other_scalar_update(
    journal: list[TunablesChanged],
) -> None:
    values = TunableValues(StrictTunables, StrictTunables(), RLock(), journal.append)
    snapshot = values.update(
        (TunableChange("reps", 3),),
        expected_revision=0,
        actor=Actor("user", "owner"),
    )
    assert snapshot.revision == 1
    assert values.capture().model == StrictTunables(reps=3)
    assert values.capture().model.points == (1.0, 2.0)
    assert len(journal) == 1
    event = journal[0]
    assert (event.revision_before, event.revision_after) == (0, 1)
    assert [(change.path, change.old, change.new) for change in event.changes] == [
        ("reps", 2, 3),
    ]


def test_strict_tuple_model_accepts_detached_json_array_replacement(
    journal: list[TunablesChanged],
) -> None:
    values = TunableValues(StrictTunables, StrictTunables(), RLock(), journal.append)
    requested: list[JsonValue] = [3.0, 4.0]
    snapshot = values.update(
        (TunableChange("points", requested),),
        expected_revision=0,
        actor=Actor("user", "owner"),
    )
    captured = values.capture()
    requested.append(99.0)
    captured.model.points = (100.0,)
    assert values.capture().model == StrictTunables(points=(3.0, 4.0))
    assert snapshot.revision == 1
    assert snapshot.values == {
        "reps": 2,
        "gate": {"lower": 2.0, "upper": 5.0},
        "points": [3.0, 4.0],
        "note": None,
    }
    assert len(journal) == 1
    event = journal[0]
    assert (event.revision_before, event.revision_after) == (0, 1)
    assert [(change.path, change.old, change.new) for change in event.changes] == [
        ("points", [1.0, 2.0], [3.0, 4.0]),
    ]


def test_strict_tuple_model_rejects_scalar_coercion_without_publication(
    journal: list[TunablesChanged],
) -> None:
    values = TunableValues(StrictTunables, StrictTunables(), RLock(), journal.append)
    with pytest.raises(ValidationError) as caught:
        values.update(
            (TunableChange("reps", "3"),),
            expected_revision=0,
            actor=Actor("user", "owner"),
        )
    assert [(error["loc"], error["type"]) for error in caught.value.errors()] == [
        (("reps",), "int_type"),
    ]
    assert values.capture().revision == 0
    assert values.capture().model == StrictTunables()
    assert journal == []


def test_tuple_replacement_and_snapshots_do_not_alias_request_or_model(
    values: TunableValues[Tunables], journal: list[TunablesChanged]
) -> None:
    requested: list[JsonValue] = [3.0, 4.0]
    result = values.update(
        (TunableChange("points", requested),),
        expected_revision=0,
        actor=Actor("agent", "planner"),
    )
    requested.append(99)
    assert values.capture().model.points == (3.0, 4.0)
    assert isinstance(result.values, dict)
    result.values["points"] = []
    assert values.capture().model.points == (3.0, 4.0)
    assert journal[0].changes[0].new == [3.0, 4.0]
    snapshot = values.snapshot()
    assert isinstance(snapshot.values, dict)
    snapshot.values["gate"] = None
    assert values.capture().model.gate == Gate()


def test_none_is_a_schema_checked_value_not_deletion(
    values: TunableValues[Tunables], journal: list[TunablesChanged]
) -> None:
    values.update(
        (TunableChange("note", "label"),),
        expected_revision=0,
        actor=Actor("user", "owner"),
    )
    values.update(
        (TunableChange("note", None),),
        expected_revision=1,
        actor=Actor("user", "owner"),
    )
    assert values.capture().model.note is None
    assert journal[-1].changes[0].old == "label"
    assert journal[-1].changes[0].new is None
    with pytest.raises(ValidationError) as caught:
        values.update(
            (TunableChange("reps", None),),
            expected_revision=2,
            actor=Actor("user", "owner"),
        )
    assert caught.value.errors()[0]["loc"] == ("reps",)
    assert values.capture().revision == 2


def test_same_value_batch_still_records_actor_and_new_revision(
    values: TunableValues[Tunables], journal: list[TunablesChanged]
) -> None:
    values.update(
        (TunableChange("reps", 2),), expected_revision=0, actor=Actor("user", "owner")
    )
    assert values.capture().revision == 1
    assert journal[0].changes[0].old == journal[0].changes[0].new == 2


def test_paths_use_model_field_names_not_aliases(
    journal: list[TunablesChanged],
) -> None:
    class Aliased(BaseModel):
        model_config = ConfigDict(extra="forbid")
        repetitions: int = Field(default=2, alias="reps")

    values = TunableValues(Aliased, Aliased(reps=2), RLock(), journal.append)
    values.update(
        (TunableChange("repetitions", 4),),
        expected_revision=0,
        actor=Actor("user", "owner"),
    )
    assert values.capture().model.repetitions == 4
    assert values.snapshot().values == {"repetitions": 4}
    with pytest.raises(ValueError, match="Unknown"):
        values.update(
            (TunableChange("reps", 5),),
            expected_revision=1,
            actor=Actor("user", "owner"),
        )


def test_optional_model_cannot_be_replaced_even_when_absent(
    journal: list[TunablesChanged],
) -> None:
    class OptionalModel(BaseModel):
        model_config = ConfigDict(extra="forbid")
        gate: Gate | None = None

    values = TunableValues(OptionalModel, OptionalModel(), RLock(), journal.append)
    with pytest.raises(ValueError, match="entire model"):
        values.update(
            (TunableChange("gate", None),),
            expected_revision=0,
            actor=Actor("user", "owner"),
        )
    with pytest.raises(ValueError, match="absent model"):
        values.update(
            (TunableChange("gate.lower", 1),),
            expected_revision=0,
            actor=Actor("user", "owner"),
        )
    assert journal == []


def test_recursive_model_paths_are_bounded_by_request_not_schema_enumeration(
    journal: list[TunablesChanged],
) -> None:
    class Recursive(BaseModel):
        model_config = ConfigDict(extra="forbid")
        reps: int = 2
        child: Recursive | None = None

    values = TunableValues(
        Recursive, Recursive(child=Recursive()), RLock(), journal.append
    )
    values.update(
        (TunableChange("child.reps", 6),),
        expected_revision=0,
        actor=Actor("user", "owner"),
    )
    assert values.snapshot().values == {"reps": 2, "child": {"reps": 6, "child": None}}
