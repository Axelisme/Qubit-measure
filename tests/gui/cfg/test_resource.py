"""Behavior of the public cfg editing and acceptance capabilities."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import pytest
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.model import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
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
from zcu_tools.gui.session.expression import evaluate_scalar_expr, validate_scalar_expr
from zcu_tools.resources.context import MetaDict


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
            values.__getitem__,
            validate_scalar_expr,
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


@dataclass
class ExpressionSources:
    captured: dict[str, object] = field(default_factory=lambda: {"device.value": -2})
    names: dict[str, object] = field(default_factory=lambda: {"offset": 1.0})
    revision: int = 0
    capture_reads: list[str] = field(default_factory=list)
    fail_read: bool = False
    fail_evaluation: bool = False

    def read(self) -> CfgResolution:
        if self.fail_read:
            raise RuntimeError("source snapshot failure")
        captured = dict(self.captured)
        md = MetaDict()
        md.update(self.names)
        fail_evaluation = self.fail_evaluation

        def evaluate(expression: str) -> int | float | complex:
            if fail_evaluation:
                raise RuntimeError("resolver defect")
            return evaluate_scalar_expr(expression, md)

        def capture(name: str) -> object:
            self.capture_reads.append(name)
            return captured[name]

        return CfgResolution(
            (SourceRevision("snapshot", CfgRevision(self.revision)),),
            evaluate,
            lambda source_id: (),
            EmptyReferences(),
            capture,
            validate_scalar_expr,
        )


def expression_defaults() -> CfgSchema:
    schema = defaults()
    schema.spec.fields["z"] = ScalarSpec("Complex", complex)
    schema.value.fields["z"] = DirectValue(0j)
    schema.spec.fields["flags"] = ScalarSpec("Flags", int)
    schema.value.fields["flags"] = DirectValue(0)
    return schema


@pytest.fixture
def captured_resource() -> tuple[CfgResource, ExpressionSources]:
    sources = ExpressionSources()
    resource = CfgResource(
        expression_defaults,
        resolution=sources.read,
        make_range=make_range,
    )
    return resource, sources


def test_capture_groups_negative_power_and_keeps_dynamic_dependencies(
    captured_resource,
) -> None:
    resource, sources = captured_resource
    changed = resource.edit(
        CfgRevision(0), (CfgEdit(("a",), EvalValue("$device.value ** 2 + offset")),)
    )
    assert resource.accept(changed.ref.revision).values["a"] == 5.0
    value = changed.tree.children["a"].value
    assert isinstance(value, EvalValue)
    assert value.expr == "(-2) ** 2 + offset"
    sources.captured["device.value"] = 100
    sources.names["offset"] = 3.0
    sources.revision += 1
    refreshed = resource.refresh(changed.ref.revision)
    assert resource.accept(refreshed.ref.revision).values["a"] == 7.0
    assert sources.capture_reads == ["device.value"]


def test_capture_uses_real_functions_complex_and_integer_semantics(
    captured_resource,
) -> None:
    resource, sources = captured_resource
    sources.captured.update({"z": 1 + 1j, "flags": 6, "angle": 0.0})
    changed = resource.edit(
        CfgRevision(0),
        (
            CfgEdit(("z",), EvalValue("$z * $z")),
            CfgEdit(("flags",), EvalValue("$flags ^ 3")),
            CfgEdit(("a",), EvalValue("cos($angle) + log(e)")),
        ),
    )
    values = resource.accept(changed.ref.revision).values
    assert values["z"] == 2j
    assert values["flags"] == 5
    assert values["a"] == 2.0
    assert sources.capture_reads.count("z") == 1
    assert isinstance(changed.tree.children["flags"].value, EvalValue)


@pytest.mark.parametrize(
    "expression",
    [
        "$",
        "$device.value +",
        "$device.",
        "$device.value[0]",
        "$device.value()",
        "$ device",
        "$device.value @ 2",
    ],
)
def test_invalid_capture_structure_rejects_batch_before_reading_sources(
    captured_resource, expression: str
) -> None:
    resource, sources = captured_resource
    before = resource.observe()
    with pytest.raises(CfgInputError, match=".") as caught:
        resource.edit(
            before.ref.revision,
            (
                CfgEdit(("literal.dot",), 10.0),
                CfgEdit(("a",), EvalValue(expression)),
            ),
        )
    assert caught.value.reason is CfgInputReason.CAPTURE_SYNTAX
    assert caught.value.edit_index == 1
    assert resource.observe() == before
    assert sources.capture_reads == []


def test_capture_missing_source_rejects_without_publishing_prefix(
    captured_resource,
) -> None:
    resource, _ = captured_resource
    before = resource.observe()
    with pytest.raises(CfgPreconditionError, match="unavailable") as caught:
        resource.edit(
            before.ref.revision,
            (
                CfgEdit(("a",), 10.0),
                CfgEdit(("literal.dot",), EvalValue("$missing")),
            ),
        )
    assert caught.value.reason is CfgPreconditionReason.CAPTURE_UNAVAILABLE
    assert caught.value.path == ("literal.dot",)
    assert caught.value.edit_index == 1
    assert resource.observe() == before


@pytest.mark.parametrize(
    "value", [True, "text", None, float("inf"), complex(1, float("nan"))]
)
def test_capture_rejects_non_literal_values(captured_resource, value: object) -> None:
    resource, sources = captured_resource
    sources.captured["device.value"] = value
    with pytest.raises(CfgInputError, match="numeric|finite") as caught:
        resource.edit(CfgRevision(0), (CfgEdit(("a",), EvalValue("$device.value")),))
    assert caught.value.reason is CfgInputReason.INVALID_VALUE
    assert resource.observe().ref.revision == 0


@pytest.mark.parametrize("expression", ["'$device.value'", "sin(", "offset +"])
def test_without_capture_tokens_incomplete_or_invalid_expression_is_saved(
    captured_resource, expression: str
) -> None:
    resource, sources = captured_resource
    changed = resource.edit(CfgRevision(0), (CfgEdit(("a",), EvalValue(expression)),))
    assert changed.status is CfgStatus.INVALID
    value = changed.tree.children["a"].value
    assert isinstance(value, EvalValue)
    assert value.expr == expression
    assert sources.capture_reads == []


def test_capture_token_boundaries_and_missing_dynamic_name(captured_resource) -> None:
    resource, sources = captured_resource
    sources.names["device_value"] = 10.0
    changed = resource.edit(
        CfgRevision(0),
        (CfgEdit(("a",), EvalValue("$device.value + device_value + missing")),),
    )
    assert changed.status is CfgStatus.INVALID
    value = changed.tree.children["a"].value
    assert isinstance(value, EvalValue)
    assert value.expr == "(-2) + device_value + missing"
    sources.names["missing"] = 1.0
    sources.captured["device.value"] = 99
    changed = resource.refresh(changed.ref.revision)
    assert resource.accept(changed.ref.revision).values["a"] == 9.0
    assert sources.capture_reads == ["device.value"]


@pytest.mark.parametrize("fault", ["source", "resolver"])
def test_refresh_fault_publishes_unavailable_without_old_resolution(
    captured_resource, fault: str
) -> None:
    resource, sources = captured_resource
    before = resource.edit(CfgRevision(0), (CfgEdit(("a",), EvalValue("offset")),))
    seen: list[CfgStatus] = []
    resource.watch(lambda observation: seen.append(observation.status))
    sources.fail_read = fault == "source"
    sources.fail_evaluation = fault == "resolver"
    sources.revision += 1
    unavailable = resource.refresh(before.ref.revision)
    assert unavailable.ref.revision == before.ref.revision + 1
    assert unavailable.status is CfgStatus.UNAVAILABLE
    assert unavailable.diagnostics[0].reason == (
        "source_failure" if fault == "source" else "resolution_failure"
    )
    value = unavailable.tree.children["a"].value
    assert isinstance(value, EvalValue)
    assert value.expr == "offset"
    assert value.resolved is None
    assert not unavailable.tree.valid
    assert not unavailable.tree.children["a"].valid
    with pytest.raises(CfgPreconditionError, match="not valid"):
        resource.accept(unavailable.ref.revision)
    assert seen == [CfgStatus.VALID, CfgStatus.UNAVAILABLE]
    old_value = before.tree.children["a"].value
    assert isinstance(old_value, EvalValue)
    assert old_value.resolved == 1.0
    sources.fail_read = False
    sources.fail_evaluation = False
    sources.names["offset"] = 8.0
    recovered = resource.refresh(unavailable.ref.revision)
    assert resource.accept(recovered.ref.revision).values["a"] == 8.0
    assert seen[-1] is CfgStatus.VALID


def test_unexpected_edit_evaluation_fault_is_not_invalid_input(
    captured_resource,
) -> None:
    resource, sources = captured_resource
    before = resource.observe()
    sources.fail_evaluation = True
    with pytest.raises(RuntimeError, match="resolver defect"):
        resource.edit(before.ref.revision, (CfgEdit(("a",), EvalValue("offset")),))
    assert resource.observe() == before


def test_unavailable_projection_drops_linked_cache_and_derived_range_result() -> None:
    shape = CfgSectionSpec(fields={"x": ScalarSpec("X", float)}, label="Shape")

    class References:
        def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
            return ("entry",)

        def resolve(self, kind: str, key: str) -> ResolvedReference:
            return ResolvedReference("Shape", CfgSectionValue({"x": DirectValue(7.0)}))

    def create_defaults() -> CfgSchema:
        return CfgSchema(
            CfgSectionSpec(
                fields={"ref": ReferenceSpec("test", [shape]), "sweep": SweepSpec()}
            ),
            CfgSectionValue(
                fields={
                    "ref": ReferenceValue(
                        "entry", CfgSectionValue({"x": DirectValue(7.0)})
                    ),
                    "sweep": SweepValue(EvalValue("frequency"), 3.0, 2),
                }
            ),
        )

    sources = Sources()
    resource = CfgResource(
        create_defaults,
        resolution=lambda: replace(sources.read(), references=References()),
        make_range=make_range,
    )
    before = resource.observe()
    assert before.status is CfgStatus.VALID
    sources.fail = True
    changed = resource.refresh(before.ref.revision)
    ref = changed.tree.children["ref"]
    assert ref.children == {}
    assert isinstance(ref.value, ReferenceValue)
    assert ref.value.chosen_key == "entry"
    assert ref.value.value.fields == {}
    assert ref.value.resolved_label is None
    sweep = changed.tree.children["sweep"].value
    assert isinstance(sweep, SweepValue)
    assert isinstance(sweep.start, EvalValue)
    assert sweep.start.expr == "frequency"
    assert sweep.start.resolved is None
    assert sweep.step == DirectValue(None)
    sources.fail = False
    recovered = resource.refresh(changed.ref.revision)
    assert recovered.status is CfgStatus.VALID
    assert recovered.tree.children["ref"].children["x"].value == DirectValue(7.0)
