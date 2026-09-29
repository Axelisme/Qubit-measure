"""Behavior of the public cfg editing and acceptance capabilities."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.model import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgInputError,
    CfgInputReason,
    CfgObservation,
    CfgPreconditionError,
    CfgPreconditionReason,
    CfgResolution,
    CfgResource,
    CfgRevision,
    CfgStaleError,
    CfgStatus,
    SourceRevision,
)


class EmptyReferences:
    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return ()

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        return None


def make_range(start: float, stop: float, *, expts: int) -> object:
    return {"start": start, "stop": stop, "expts": expts}


@dataclass
class Sources:
    revision: int = 0
    values: dict[str, float] = field(default_factory=lambda: {"frequency": 2.0})
    reads: int = 0
    fail: bool = False

    def read(self) -> CfgResolution:
        self.reads += 1
        if self.fail:
            raise RuntimeError("source snapshot failure")
        values = dict(self.values)

        def evaluate(expression: str) -> float:
            return values[expression]

        return CfgResolution(
            (SourceRevision("md", CfgRevision(self.revision)),),
            evaluate,
            lambda source_id: (),
            EmptyReferences(),
        )


def defaults() -> CfgSchema:
    return CfgSchema(
        CfgSectionSpec(
            fields={
                "a": ScalarSpec("A", float),
                "literal.dot": ScalarSpec("B", float),
            }
        ),
        CfgSectionValue(
            fields={"a": DirectValue(1.0), "literal.dot": DirectValue(2.0)}
        ),
    )


def make_resource(sources: Sources | None = None) -> CfgResource:
    source = sources or Sources()
    return CfgResource(defaults, resolution=source.read, make_range=make_range)


def test_batch_is_atomic_and_literal_path_is_not_split() -> None:
    resource = make_resource()
    before = resource.observe()
    with pytest.raises(CfgInputError, match="unknown field") as caught:
        resource.edit(
            before.ref.revision,
            (
                CfgEdit(("a",), 7.0),
                CfgEdit(("missing",), 2.0),
            ),
        )
    assert caught.value.path == ("missing",)
    assert caught.value.edit_index == 1
    assert caught.value.reason is CfgInputReason.UNKNOWN_PATH
    assert resource.observe() == before
    changed = resource.edit(before.ref.revision, (CfgEdit(("literal.dot",), 3.0),))
    assert resource.accept(changed.ref.revision).values["literal.dot"] == 3.0


def test_success_same_value_and_empty_batch_each_publish() -> None:
    resource = make_resource()
    for expected, edits in enumerate(
        [
            (CfgEdit(("a",), 8.0),),
            (CfgEdit(("a",), 8.0),),
            (),
        ],
        start=1,
    ):
        observation = resource.edit(CfgRevision(expected - 1), edits)
        assert observation.ref.revision == expected
    observation = resource.observe()
    with pytest.raises(CfgStaleError, match="Expected cfg revision") as caught:
        resource.edit(CfgRevision(0), ())
    assert caught.value.expected.revision == 0
    assert caught.value.actual == observation.ref
    assert resource.observe().ref == observation.ref


def test_incomplete_text_is_published_invalid_and_blocks_acceptance() -> None:
    resource = make_resource()
    changed = resource.edit(CfgRevision(0), (CfgEdit(("a",), DirectValue(raw="1e-")),))
    assert changed.status is CfgStatus.INVALID
    assert changed.diagnostics[0].path == ("a",)
    assert changed.tree.children["a"].value == DirectValue(
        None, raw="1e-", error="could not convert string to float: '1e-'"
    )
    with pytest.raises(CfgPreconditionError, match="not valid") as caught:
        resource.accept(changed.ref.revision)
    assert caught.value.reason is CfgPreconditionReason.NOT_VALID


def test_observe_and_accept_never_read_sources_and_accept_is_detached() -> None:
    sources = Sources()
    resource = make_resource(sources)
    changed = resource.edit(CfgRevision(0), (CfgEdit(("a",), EvalValue("frequency")),))
    reads = sources.reads
    accepted = resource.accept(changed.ref.revision)
    assert accepted.values["a"] == 2.0
    sources.values["frequency"] = 10.0
    sources.revision = 1
    changed.tree.children.clear()
    assert resource.observe().tree.children
    accepted.values["a"] = 100.0
    assert resource.accept(accepted.ref.revision).values["a"] == 2.0
    assert sources.reads == reads
    refreshed = resource.refresh(accepted.ref.revision)
    assert resource.accept(refreshed.ref.revision).values["a"] == 10.0
    assert accepted.source_basis[0].revision == 0
    assert refreshed.source_basis[0].revision == 1


def test_watch_initial_order_fault_isolation_and_reentry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resource = make_resource()
    seen: list[int] = []
    rejected: list[CfgPreconditionReason] = []

    def reenter(observation: CfgObservation) -> None:
        for action in (
            lambda: resource.edit(observation.ref.revision, ()),
            lambda: resource.reset(observation.ref.revision),
            lambda: resource.refresh(observation.ref.revision),
            lambda: resource.accept(observation.ref.revision),
            resource.revoke,
        ):
            with pytest.raises(CfgPreconditionError, match="forbidden") as caught:
                action()
            rejected.append(caught.value.reason)
        observation.tree.children.clear()
        raise RuntimeError("subscriber bug")

    resource.watch(reenter)
    unsubscribe = resource.watch(
        lambda observation: seen.append(observation.ref.revision)
    )
    changed = resource.edit(CfgRevision(0), ())
    assert seen == [0, 1]
    assert len(rejected) == 10
    assert set(rejected) == {CfgPreconditionReason.REENTRANT_MUTATION}
    assert changed.tree.children
    assert "subscriber bug" in caplog.text
    unsubscribe()
    unsubscribe()
    resource.edit(CfgRevision(1), ())
    assert seen == [0, 1]


def test_reset_failure_keeps_publication_and_does_not_notify() -> None:
    fail = False

    def create_defaults() -> CfgSchema:
        if fail:
            raise RuntimeError("defaults failure")
        return defaults()

    resource = CfgResource(
        create_defaults, resolution=Sources().read, make_range=make_range
    )
    seen: list[int] = []
    resource.watch(lambda observation: seen.append(observation.ref.revision))
    before = resource.edit(CfgRevision(0), (CfgEdit(("a",), 9.0),))
    fail = True
    with pytest.raises(RuntimeError, match="defaults failure"):
        resource.reset(before.ref.revision)
    assert resource.observe() == before
    assert seen == [0, 1]
    fail = False
    reset = resource.reset(before.ref.revision)
    assert resource.accept(reset.ref.revision).values["a"] == 1.0


def test_busy_policy_is_owned_by_resource_and_acceptance_is_separate() -> None:
    resource = CfgResource(
        defaults,
        resolution=Sources().read,
        make_range=make_range,
        mutation_allowed=lambda: False,
    )
    before = resource.observe()
    with pytest.raises(CfgPreconditionError, match="blocked") as caught:
        resource.edit(before.ref.revision, ())
    assert caught.value.reason is CfgPreconditionReason.MUTATION_BLOCKED
    assert resource.accept(before.ref.revision).values["a"] == 1.0


def test_revocation_invalidates_handles_and_never_reuses_identity() -> None:
    resource = make_resource()
    before = resource.observe()
    unsubscribe = resource.watch(lambda observation: None)
    resource.revoke()
    unsubscribe()
    resource.revoke()
    for action in (
        resource.observe,
        lambda: resource.watch(lambda observation: None),
        lambda: resource.edit(before.ref.revision, ()),
        lambda: resource.accept(before.ref.revision),
    ):
        with pytest.raises(CfgPreconditionError, match="revoked") as caught:
            action()
        assert caught.value.reason is CfgPreconditionReason.RESOURCE_GONE
    assert make_resource().observe().ref.cfg_id != before.ref.cfg_id
