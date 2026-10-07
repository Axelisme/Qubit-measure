"""Opaque expression audit through the public Engine and persisted journal."""

from collections import UserDict
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from zcu_tools.experiment.workflows import (
    Actor,
    DeviceSnapshot,
    Done,
    Next,
    RevisionConflict,
    RunIdentity,
    RunPaths,
    Step,
    TunableChange,
    WorkflowEnv,
    workflow,
)

from ._engine_fakes import START, Plan, Rig, State, Tunables


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    low: int = Field(default=1, ge=1)
    high: int = Field(default=10, ge=1)


class NestedKnobs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = Field(default=1, ge=1)
    limits: Limits = Field(default_factory=Limits)
    levels: list[int] = Field(default_factory=lambda: [1, 2])


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def points(
    env: WorkflowEnv[int], plan: Plan, tunables: Tunables, state: State
) -> Step[State, int]:
    """Commit each captured value without interpreting any expression text."""
    if state.index == plan.count:
        return Done()
    yield from env.wait_until(START + timedelta(seconds=1))
    state.index += 1
    return Next(tunables.value, state)


def test_start_detaches_opaque_originals_and_persists_revision_zero(rig: Rig) -> None:
    originals = {"value": "not_a_valid_expression(@)"}
    rig.start(points, tunables_originals=originals)
    originals["value"] = "changed_by_host"
    snapshot = rig.engine.tunables("run")
    assert snapshot.revision == 0
    assert snapshot.values == {"value": 1}
    assert snapshot.originals == {"value": "not_a_valid_expression(@)"}
    startup = rig.events()[0]
    assert startup["kind"] == "run_started"
    assert startup["tunables"] == {"value": 1}
    assert startup["tunables_originals"] == dict(snapshot.originals)
    assert rig.engine.execute("run").lifecycle == "done"
    assert [record.encoded_record.value for record in rig.engine.records("run")] == [
        1,
        1,
    ]


@pytest.mark.parametrize("originals", [None, {}])
def test_start_without_originals_preserves_ordinary_value_behavior(
    rig: Rig, originals: Mapping[str, str] | None
) -> None:
    rig.start(points, tunables_originals=originals)
    snapshot = rig.engine.tunables("run")
    assert (snapshot.revision, snapshot.values, snapshot.originals) == (
        0,
        {"value": 1},
        {},
    )
    assert rig.events()[0]["tunables_originals"] == {}
    assert rig.engine.execute("run").lifecycle == "done"


def test_update_audits_old_and_new_values_and_originals_at_one_revision(
    rig: Rig,
) -> None:
    rig.start(points, tunables_originals={"value": "md.start"})
    before = rig.engine.tunables("run")
    changed = rig.engine.update_tunables(
        "run",
        (TunableChange("value", 8, original="md.next + 2"),),
        expected_revision=0,
        actor=Actor("agent", "calibration"),
    )
    assert (changed.revision, changed.values, changed.originals) == (
        1,
        {"value": 8},
        {"value": "md.next + 2"},
    )
    assert rig.engine.tunables("run") == changed
    assert before.values == {"value": 1}
    assert before.originals == {"value": "md.start"}
    event = rig.events()[-1]
    assert event["kind"] == "tunables_changed"
    assert (event["revision_before"], event["revision_after"]) == (0, 1)
    assert event["actor"] == {"kind": "agent", "name": "calibration"}
    assert event["time"] == START.isoformat()
    assert event["changes"] == [
        {
            "path": "value",
            "old": 1,
            "new": 8,
            "old_original": "md.start",
            "new_original": "md.next + 2",
        }
    ]


def test_same_value_original_edit_is_a_revision_and_omission_clears_text(
    rig: Rig,
) -> None:
    rig.start(points, tunables_originals={"value": "md.one"})
    edited = rig.engine.update_tunables(
        "run",
        (TunableChange("value", 1, original="md.other_one"),),
        expected_revision=0,
        actor=Actor("user", "operator"),
    )
    assert edited.revision == 1
    assert edited.values == {"value": 1}
    assert edited.originals == {"value": "md.other_one"}
    cleared = rig.engine.update_tunables(
        "run",
        (TunableChange("value", 1),),
        expected_revision=1,
        actor=Actor("user", "operator"),
    )
    assert (cleared.revision, cleared.values, cleared.originals) == (
        2,
        {"value": 1},
        {},
    )
    assert rig.events()[-1]["changes"] == [
        {
            "path": "value",
            "old": 1,
            "new": 1,
            "old_original": "md.other_one",
            "new_original": None,
        }
    ]
    assert edited.originals == {"value": "md.other_one"}


@pytest.mark.parametrize("path", ["", "missing", "value.child", "value.0"])
def test_invalid_start_original_path_creates_no_files_and_keeps_prior_run(
    rig: Rig, path: str
) -> None:
    rig.start(points, tunables_originals={"value": "md.one"})
    previous = rig.engine.execute("run")
    paths = RunPaths(rig.root / "next-metadata", rig.root / "next-data")

    def replacement(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        return points(env, plan, tun, state)

    declared = workflow(
        "next",
        plan=Plan,
        tunables=Tunables,
        state=State,
        record=int,
        init=rig.init,
    )(replacement)
    with pytest.raises(ValueError, match="path|field|model|leaf|Unknown"):
        rig.engine.start(
            declared,
            plan=Plan(),
            tunables=Tunables(),
            tunables_originals={path: "opaque"},
            paths=paths,
            identity=RunIdentity("next", DeviceSnapshot({}), "offline"),
        )
    assert rig.engine.status() == previous
    assert rig.engine.tunables("run").originals == {"value": "md.one"}
    assert not paths.metadata_root.exists()
    assert not paths.data_root.exists()


def test_stale_or_invalid_candidate_publishes_neither_values_nor_originals(
    rig: Rig,
) -> None:
    rig.start(points, tunables_originals={"value": "md.one"})
    before = rig.engine.tunables("run")
    events = rig.events()
    with pytest.raises(RevisionConflict):
        rig.engine.update_tunables(
            "run",
            (TunableChange("value", 2, original="md.two"),),
            expected_revision=9,
            actor=Actor("agent", "stale"),
        )
    with pytest.raises(ValidationError):
        rig.engine.update_tunables(
            "run",
            (TunableChange("value", 0, original="md.zero"),),
            expected_revision=0,
            actor=Actor("user", "invalid"),
        )
    with pytest.raises(ValueError, match="path|field|Unknown"):
        rig.engine.update_tunables(
            "run",
            (TunableChange("missing", 2, original="md.two"),),
            expected_revision=0,
            actor=Actor("user", "unknown"),
        )
    assert rig.engine.tunables("run") == before
    assert rig.events() == events
    status = rig.engine.status()
    assert status is not None and status.lifecycle == "running"


def test_nested_leaf_and_whole_sequence_originals_share_atomic_batch(rig: Rig) -> None:
    def nested(
        env: WorkflowEnv[int], plan: Plan, tun: NestedKnobs, state: State
    ) -> Step[State, int]:
        if state.index == plan.count:
            return Done()
        yield from env.wait_until(START)
        state.index += 1
        return Next(tun.value, state)

    declared = workflow(
        "nested",
        plan=Plan,
        tunables=NestedKnobs,
        state=State,
        record=int,
        init=rig.init,
    )(nested)
    rig.engine.start(
        declared,
        plan=Plan(),
        tunables=NestedKnobs(),
        tunables_originals={
            "value": "md.one",
            "limits.low": "md.low",
            "levels": "opaque_sequence",
        },
        paths=rig.paths,
        identity=RunIdentity("run", DeviceSnapshot({}), "offline"),
    )
    before = rig.engine.tunables("run")
    events = rig.events()
    with pytest.raises(ValidationError):
        rig.engine.update_tunables(
            "run",
            (
                TunableChange("limits.low", 3, original="md.three"),
                TunableChange("value", 0, original="md.zero"),
            ),
            expected_revision=0,
            actor=Actor("user", "invalid-batch"),
        )
    assert rig.engine.tunables("run") == before
    assert rig.events() == events
    changed = rig.engine.update_tunables(
        "run",
        (
            TunableChange("limits.low", 3, original="md.three"),
            TunableChange("levels", [4, 5]),
        ),
        expected_revision=0,
        actor=Actor("agent", "batch"),
    )
    assert changed.revision == 1
    assert changed.values == {
        "value": 1,
        "limits": {"low": 3, "high": 10},
        "levels": [4, 5],
    }
    assert changed.originals == {"value": "md.one", "limits.low": "md.three"}
    assert before.originals == {
        "value": "md.one",
        "limits.low": "md.low",
        "levels": "opaque_sequence",
    }
    for illegal in ("limits", "levels.0"):
        with pytest.raises(ValueError, match="path|field|model|leaf|Unknown"):
            rig.engine.update_tunables(
                "run",
                (TunableChange(illegal, 7, original="invalid"),),
                expected_revision=1,
                actor=Actor("user", "bad-path"),
            )
    assert rig.engine.tunables("run") == changed


def test_non_text_start_original_is_rejected_before_startup_io(rig: Rig) -> None:
    originals = UserDict({"value": "text"})
    # Model an untyped caller bypassing the mapping's static value annotation.
    object.__setattr__(originals, "data", {"value": 42})
    with pytest.raises(ValueError, match="original.*(str|text)"):
        rig.start(points, tunables_originals=originals)
    assert rig.engine.status() is None
    assert not rig.paths.metadata_root.exists()
    assert not rig.paths.data_root.exists()


def test_non_text_update_original_is_rejected_at_engine_boundary(rig: Rig) -> None:
    rig.start(points, tunables_originals={"value": "md.one"})
    before = rig.engine.tunables("run")
    events = rig.events()
    change = TunableChange("value", 2, original="text")
    # Model an untyped caller bypassing the dataclass's static field annotation.
    object.__setattr__(change, "original", 42)
    with pytest.raises(ValueError, match="original.*(str|text)"):
        rig.engine.update_tunables(
            "run",
            (change,),
            expected_revision=0,
            actor=Actor("agent", "malformed"),
        )
    assert rig.engine.tunables("run") == before
    assert rig.events() == events


def test_journal_failure_retains_both_previous_values_and_originals(rig: Rig) -> None:
    rig.start(points, tunables_originals={"value": "md.one"})
    before = rig.engine.tunables("run")
    journal = rig.paths.metadata_root / "journal.jsonl"
    journal.unlink()
    journal.mkdir()
    with pytest.raises(IsADirectoryError):
        rig.engine.update_tunables(
            "run",
            (TunableChange("value", 2, original="md.two"),),
            expected_revision=0,
            actor=Actor("user", "io-failure"),
        )
    assert rig.engine.tunables("run") == before
    status = rig.engine.status()
    assert status is not None and status.lifecycle == "failed"
    assert status.revision == 0


def test_current_step_keeps_capture_and_next_step_uses_new_revision(rig: Rig) -> None:
    seen: list[int] = []

    def step(
        env: WorkflowEnv[int], plan: Plan, tunables: Tunables, state: State
    ) -> Step[State, int]:
        if state.index == plan.count:
            return Done()
        seen.append(tunables.value)
        if state.index == 0:
            changed = rig.engine.update_tunables(
                "run",
                (TunableChange("value", 3, original="md.three"),),
                expected_revision=0,
                actor=Actor("agent", "during-step"),
            )
            assert changed.originals == {"value": "md.three"}
        yield from env.wait_until(START)
        state.index += 1
        return Next(tunables.value, state)

    rig.start(step, tunables_originals={"value": "md.one"})
    assert rig.engine.execute("run").lifecycle == "done"
    assert seen == [1, 3]
    assert [
        (record.revision, record.encoded_record.value)
        for record in rig.engine.records("run")
    ] == [
        (0, 1),
        (1, 3),
    ]


def test_resume_repeats_discarded_step_with_latest_values_and_originals(
    rig: Rig,
) -> None:
    rig.start(points, tunables_originals={"value": "md.one"})
    rig.clock.on_wait = lambda _signal: rig.engine.pause("run")
    assert rig.engine.execute("run").lifecycle == "paused"
    assert rig.engine.records("run") == ()
    rig.engine.update_tunables(
        "run",
        (TunableChange("value", 4, original="md.four"),),
        expected_revision=0,
        actor=Actor("agent", "paused"),
    )
    rig.clock.on_wait = None
    assert rig.engine.resume("run", devices=DeviceSnapshot({})).lifecycle == "running"
    assert rig.engine.execute("run").lifecycle == "done"
    assert rig.engine.tunables("run").originals == {"value": "md.four"}
    assert [
        (record.revision, record.encoded_record.value)
        for record in rig.engine.records("run")
    ] == [
        (1, 4),
        (1, 4),
    ]
