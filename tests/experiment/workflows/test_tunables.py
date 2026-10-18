"""Whole native model updates through the public Engine and journal seams."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    Next,
    RevisionConflict,
    RunIdentity,
    Step,
    WorkflowEnv,
    workflow,
)

from ._engine_fakes import Plan, Rig, State


class Gate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    lower: float = 2
    upper: float = 5

    @model_validator(mode="after")
    def ordered(self) -> "Gate":
        if self.lower > self.upper:
            raise ValueError("lower must not exceed upper")
        return self


class Tunables(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reps: int = Field(default=2, ge=1)
    gate: Gate | None = Field(default_factory=Gate)
    points: tuple[float, ...] = (1.0, 2.0)
    gates: list[Gate] = Field(default_factory=lambda: [Gate()])
    note: str | None = None


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    rig = Rig(tmp_path)

    @workflow("whole-model", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index == plan.count:
            return Done()
        yield from env.wait_until(env.started_at)
        state.index += 1
        return Next(tun.reps, state)

    rig.engine.start(
        step,
        plan=Plan(),
        tunables=Tunables(),
        paths=rig.paths,
        identity=RunIdentity(
            "run",
            SoftwareProvenance(
                software_versions={},
                git_commit=None,
                git_dirty=None,
                qick_version=None,
                soc_fingerprint=None,
                hostname="offline",
            ),
        ),
    )
    return rig


def test_complete_nested_model_list_replacement_and_detached_journal(rig: Rig) -> None:
    candidate = Tunables(
        reps=3, gate=None, points=(3.0,), gates=[Gate(lower=6, upper=7)], note="new"
    )
    result = rig.engine.update_tunables(
        "run", tunables=candidate, expected_revision=0, actor=Actor("agent", "planner")
    )
    expected = {
        "reps": 3,
        "gate": None,
        "points": [3.0],
        "gates": [{"lower": 6.0, "upper": 7.0}],
        "note": "new",
    }
    assert result.revision == 1 and result.values == expected
    event = rig.events()[-1]
    assert event["actor"] == {"kind": "agent", "name": "planner"}
    assert (event["revision_before"], event["revision_after"]) == (0, 1)
    assert event["tunables"] == expected
    assert rig.kinds() == ("run_started", "tunables_changed")
    candidate.gates[0].lower = -100
    assert isinstance(result.values, dict)
    result.values["gates"] = []
    assert rig.engine.tunables("run").values == expected
    assert rig.events()[-1]["tunables"] == expected


def test_same_complete_values_still_append_and_increment(rig: Rig) -> None:
    for revision in range(2):
        result = rig.engine.update_tunables(
            "run", Tunables(), expected_revision=revision, actor=Actor("user", "gui")
        )
        assert result.revision == revision + 1
    assert rig.kinds() == ("run_started", "tunables_changed", "tunables_changed")
    assert [event["tunables"] for event in rig.events()[1:]] == [
        Tunables().model_dump(mode="json"),
        Tunables().model_dump(mode="json"),
    ]


@pytest.mark.parametrize(
    ("candidate", "location"),
    [
        (Tunables.model_construct(reps=0), ("reps",)),
        (Tunables.model_construct(reps="3"), ("reps",)),
        (
            Tunables.model_construct(gates=[Gate.model_construct(lower=7, upper=2)]),
            ("gates", 0),
        ),
        (Tunables(points=(float("inf"),)), ("points", 0)),
        (Tunables(gates=[Gate(lower=float("nan"))]), ("gates", 0, "lower")),
    ],
)
def test_revalidate_complete_native_model_without_publication(
    rig: Rig, candidate: Tunables, location: tuple[str | int, ...]
) -> None:
    before = rig.engine.tunables("run")
    with pytest.raises(ValidationError) as caught:
        rig.engine.update_tunables(
            "run", candidate, expected_revision=0, actor=Actor("user", "gui")
        )
    assert caught.value.errors()[0]["loc"] == location
    assert rig.engine.tunables("run") == before
    assert rig.kinds() == ("run_started",)


def test_wrong_declared_model_is_rejected_without_publication(rig: Rig) -> None:
    with pytest.raises(TypeError, match="declared tunables"):
        rig.engine.update_tunables(
            "run", Plan(), expected_revision=0, actor=Actor("user", "gui")
        )
    assert rig.engine.tunables("run").revision == 0
    assert rig.kinds() == ("run_started",)


def test_stale_complete_model_keeps_previous_revision_and_journal(rig: Rig) -> None:
    rig.engine.update_tunables(
        "run", Tunables(reps=3), expected_revision=0, actor=Actor("user", "gui")
    )
    before = rig.engine.tunables("run")
    with pytest.raises(RevisionConflict) as caught:
        rig.engine.update_tunables(
            "run", Tunables(reps=4), expected_revision=0, actor=Actor("agent", "rpc")
        )
    assert (caught.value.expected, caught.value.actual) == (0, 1)
    assert rig.engine.tunables("run") == before
    assert len(rig.events()) == 2


def test_append_failure_keeps_complete_model_revision_and_original_cause(
    rig: Rig,
) -> None:
    journal = rig.paths.metadata_root / "journal.jsonl"
    original = journal.read_bytes()
    before = rig.engine.tunables("run")
    journal.unlink()
    journal.mkdir()
    try:
        with pytest.raises(IsADirectoryError) as caught:
            rig.engine.update_tunables(
                "run", Tunables(reps=4), expected_revision=0, actor=Actor("user", "gui")
            )
        assert caught.value.filename == str(journal)
        assert rig.engine.tunables("run") == before
        status = rig.engine.status()
        assert status is not None and status.lifecycle == "failed"
        assert status.reason is not None and str(journal) in status.reason
    finally:
        journal.rmdir()
        journal.write_bytes(original)


def test_current_step_keeps_capture_and_next_step_uses_whole_replacement(
    rig: Rig,
) -> None:
    def update(_signal: object) -> None:
        # The wait port is reached only after this invocation captured revision zero.
        rig.clock.on_wait = None
        result = rig.engine.update_tunables(
            "run",
            Tunables(reps=8, gates=[]),
            expected_revision=0,
            actor=Actor("agent", "rpc"),
        )
        assert result.revision == 1

    rig.clock.on_wait = update
    assert rig.engine.execute("run").lifecycle == "done"
    assert [
        (r.revision, r.encoded_record.value) for r in rig.engine.records("run")
    ] == [(0, 2), (1, 8)]
    first = (rig.paths.data_root / "iter/000001/record.json").read_text()
    assert '"reps":2' in first.replace(" ", "")
