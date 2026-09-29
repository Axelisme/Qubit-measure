"""Behavior of the public cfg editing and acceptance capabilities."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import permutations

import pytest
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.model import (
    CenteredSweepSpec,
    CenteredSweepValue,
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


def test_restore_skips_defaults_until_reset_and_detaches_input_snapshot() -> None:
    calls: list[str] = []

    def fresh_defaults() -> CfgSchema:
        calls.append("defaults")
        return defaults()

    initial = defaults()
    initial.value.fields["a"] = DirectValue(7.0)
    source = Sources()
    resource = CfgResource(
        fresh_defaults, initial=initial, resolution=source.read, make_range=make_range
    )
    before = resource.observe()
    initial.value.fields["a"] = DirectValue(99.0)
    assert resource.accept(before.ref.revision).values["a"] == 7.0
    reads = source.reads
    snapshot = resource.snapshot_inputs()
    snapshot.value.fields["a"] = DirectValue(99.0)
    assert resource.snapshot_inputs().value.fields["a"] == DirectValue(7.0)
    assert source.reads == reads
    assert calls == []
    reset = resource.reset(before.ref.revision)
    assert reset.ref.cfg_id == before.ref.cfg_id
    assert resource.accept(reset.ref.revision).values["a"] == 1.0
    assert calls == ["defaults"]


def test_owner_replacement_publishes_incomplete_input_once_without_new_identity() -> (
    None
):
    resource = make_resource()
    before = resource.observe()
    seen: list[CfgObservation] = []
    resource.watch(seen.append)
    schema = defaults()
    schema.value.fields["a"] = DirectValue(None, raw="-")
    changed = resource.replace_inputs(before.ref.revision, schema)
    assert changed.status is CfgStatus.INVALID
    assert changed.ref.cfg_id == before.ref.cfg_id
    assert changed.ref.revision == before.ref.revision + 1
    assert len(seen) == 2
    schema.value.fields["a"] = DirectValue(9.0)
    assert resource.snapshot_inputs().value.fields["a"] == DirectValue(None, raw="-")
    with pytest.raises(CfgPreconditionError) as caught:
        resource.accept(changed.ref.revision)
    assert caught.value.reason is CfgPreconditionReason.NOT_VALID


@pytest.mark.parametrize("fault", ["definition", "source"])
def test_owner_replacement_failure_preserves_publication_and_inputs(fault: str) -> None:
    source = Sources()
    resource = make_resource(source)
    before = resource.observe()
    inputs = resource.snapshot_inputs()
    seen: list[CfgObservation] = []
    resource.watch(seen.append)
    candidate = defaults()
    candidate.value.fields["a"] = DirectValue(8.0)
    if fault == "definition":
        candidate.spec = CfgSectionSpec(fields={"a": ScalarSpec("Different", float)})
    else:
        source.fail = True
    with pytest.raises(RuntimeError, match="definition|source snapshot failure"):
        resource.replace_inputs(before.ref.revision, candidate)
    assert resource.observe() == before
    assert resource.snapshot_inputs() == inputs
    assert len(seen) == 1


def test_owner_replacement_obeys_revision_busy_reentry_and_revocation() -> None:
    allowed = True
    resource = CfgResource(
        defaults,
        resolution=Sources().read,
        make_range=make_range,
        mutation_allowed=lambda: allowed,
    )
    before = resource.observe()
    with pytest.raises(CfgStaleError):
        resource.replace_inputs(CfgRevision(before.ref.revision + 1), defaults())
    allowed = False
    with pytest.raises(CfgPreconditionError) as busy:
        resource.replace_inputs(before.ref.revision, defaults())
    assert busy.value.reason is CfgPreconditionReason.MUTATION_BLOCKED
    allowed = True
    errors: list[CfgPreconditionReason] = []

    def reenter(observation: CfgObservation) -> None:
        with pytest.raises(CfgPreconditionError) as caught:
            resource.replace_inputs(observation.ref.revision, defaults())
        errors.append(caught.value.reason)

    unsubscribe = resource.watch(reenter)
    assert errors == [CfgPreconditionReason.REENTRANT_MUTATION]
    unsubscribe()
    resource.revoke()
    for operation in (
        resource.snapshot_inputs,
        lambda: resource.replace_inputs(before.ref.revision, defaults()),
    ):
        with pytest.raises(CfgPreconditionError) as gone:
            operation()
        assert gone.value.reason is CfgPreconditionReason.RESOURCE_GONE


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


@pytest.mark.parametrize("failed", [False, True])
def test_source_refresh_is_not_blocked_by_manual_edit_policy(failed: bool) -> None:
    sources = Sources()
    schema = defaults()
    schema.value.fields["a"] = EvalValue("frequency")
    resource = CfgResource(
        defaults,
        initial=schema,
        resolution=sources.read,
        make_range=make_range,
        mutation_allowed=lambda: False,
    )
    before = resource.observe()
    accepted = resource.accept(before.ref.revision)
    sources.values["frequency"] = 7.0
    sources.revision += 1
    sources.fail = failed
    changed = resource.refresh(before.ref.revision)
    assert changed.ref.revision == before.ref.revision + 1
    assert changed.status is (CfgStatus.UNAVAILABLE if failed else CfgStatus.VALID)
    assert accepted.values["a"] == 2.0
    if not failed:
        assert resource.accept(changed.ref.revision).values["a"] == 7.0
    with pytest.raises(CfgPreconditionError) as blocked:
        resource.edit(changed.ref.revision, ())
    assert blocked.value.reason is CfgPreconditionReason.MUTATION_BLOCKED


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
    evaluation_reads: list[str] = field(default_factory=list)

    def read(self) -> CfgResolution:
        if self.fail_read:
            raise RuntimeError("source snapshot failure")
        captured = dict(self.captured)
        md = MetaDict()
        md.update(self.names)
        fail_evaluation = self.fail_evaluation

        def evaluate(expression: str) -> int | float | complex:
            self.evaluation_reads.append(expression)
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


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("step_first", [False, True])
def test_separate_range_edits_preserve_command_order(
    centered: bool, step_first: bool
) -> None:
    spec = CenteredSweepSpec() if centered else SweepSpec()
    value = CenteredSweepValue(1.0, 2.0, 3) if centered else SweepValue(0.0, 2.0, 3)
    schema = CfgSchema(
        CfgSectionSpec(fields={"range": spec}), CfgSectionValue({"range": value})
    )
    resource = CfgResource(
        lambda: schema, resolution=Sources().read, make_range=make_range
    )
    step = CfgEdit(("range", "step"), 0.5)
    geometry = CfgEdit(("range", "span" if centered else "stop"), 4.0)
    edits = (step, geometry) if step_first else (geometry, step)
    changed = resource.edit(resource.observe().ref.revision, edits)
    assert resource.accept(changed.ref.revision).values["range"] == {
        "start": -1.0 if centered else 0.0,
        "stop": 3.0 if centered else 4.0,
        "expts": 5 if step_first else 9,
    }


def test_range_intent_preserves_unexpected_source_fault() -> None:
    sources = Sources()
    broken = False
    defect = ValueError("range source defect")

    def resolution() -> CfgResolution:
        fail = broken
        source = sources.read()

        def evaluate(expression: str) -> int | float | complex:
            if fail:
                raise defect
            return source.evaluate_expression(expression)

        return replace(source, evaluate_expression=evaluate)

    schema = CfgSchema(
        CfgSectionSpec(fields={"range": SweepSpec()}),
        CfgSectionValue({"range": SweepValue(EvalValue("frequency"), 6.0, 3)}),
    )
    resource = CfgResource(lambda: schema, resolution=resolution, make_range=make_range)
    before = resource.observe()
    broken = True
    with pytest.raises(ValueError, match="range source defect") as caught:
        resource.edit(before.ref.revision, (CfgEdit(("range", "step"), 1.0),))
    assert caught.value is defect
    assert resource.observe() == before


def test_replacing_failed_expression_does_not_evaluate_old_input(
    captured_resource,
) -> None:
    resource, sources = captured_resource
    changed = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("a",), EvalValue("offset")),)
    )
    sources.fail_evaluation = True
    unavailable = resource.refresh(changed.ref.revision)
    assert unavailable.status is CfgStatus.UNAVAILABLE
    sources.evaluation_reads.clear()
    recovered = resource.edit(unavailable.ref.revision, (CfgEdit(("a",), 7.0),))
    assert recovered.status is CfgStatus.VALID
    assert resource.accept(recovered.ref.revision).values["a"] == 7.0
    assert sources.evaluation_reads == []


def test_replaced_intermediate_expression_is_not_evaluated(captured_resource) -> None:
    resource, sources = captured_resource
    sources.fail_evaluation = True
    changed = resource.edit(
        resource.observe().ref.revision,
        (
            CfgEdit(("a",), EvalValue("offset")),
            CfgEdit(("a",), 7.0),
        ),
    )
    assert changed.status is CfgStatus.VALID
    assert resource.accept(changed.ref.revision).values["a"] == 7.0
    assert sources.evaluation_reads == []


def test_retained_expression_uses_new_source_basis(captured_resource) -> None:
    resource, sources = captured_resource
    changed = resource.edit(
        resource.observe().ref.revision, (CfgEdit(("a",), EvalValue("offset * 2")),)
    )
    assert resource.accept(changed.ref.revision).values["a"] == 2.0
    sources.names["offset"] = 6.0
    sources.revision += 1
    newer = resource.edit(changed.ref.revision, (CfgEdit(("literal.dot",), 7.0),))
    assert resource.accept(newer.ref.revision).values["a"] == 12.0
    assert newer.source_basis[0].revision == sources.revision


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


@pytest.mark.parametrize("path", ["a", ["a"], ("",), (1,), (None,)])
def test_edit_rejects_malformed_python_path(path) -> None:
    with pytest.raises(CfgInputError) as caught:
        CfgEdit(path, 2.0)
    assert caught.value.reason is CfgInputReason.MALFORMED_INPUT


@pytest.mark.parametrize("edits", [None, [], [CfgEdit(("a",), 2.0)], (None,)])
def test_edit_rejects_malformed_batch_without_reading_sources(edits) -> None:
    sources = Sources()
    resource = make_resource(sources)
    before = resource.observe()
    reads = sources.reads
    with pytest.raises(CfgInputError) as caught:
        resource.edit(before.ref.revision, edits)
    assert caught.value.reason is CfgInputReason.MALFORMED_INPUT
    assert resource.observe() == before
    assert sources.reads == reads


@pytest.mark.parametrize("revision", [True, -1, 0.0, "0"])
def test_invalid_revision_never_reads_sources_or_publishes(revision) -> None:
    sources = Sources()
    resource = make_resource(sources)
    before = resource.observe()
    reads = sources.reads
    with pytest.raises(CfgInputError) as caught:
        resource.edit(revision, ())
    assert caught.value.reason is CfgInputReason.MALFORMED_INPUT
    assert resource.observe() == before
    assert sources.reads == reads


@pytest.mark.parametrize("key", ["", "__expr", "__ref", 1])
@pytest.mark.parametrize("nested", [False, True])
def test_definition_rejects_unaddressable_or_reserved_keys(key, nested: bool) -> None:
    shape = CfgSectionSpec(fields={key: ScalarSpec("A", float)})
    spec = (
        CfgSectionSpec(fields={"ref": ReferenceSpec("test", [shape])})
        if nested
        else shape
    )
    sources = Sources()
    with pytest.raises(ValueError, match="field name"):
        CfgResource(
            lambda: CfgSchema(spec, CfgSectionValue({})),
            resolution=sources.read,
            make_range=make_range,
        )
    assert sources.reads == 0


@pytest.mark.parametrize("failure_call", [1, 2])
def test_unexpected_capture_validator_fault_preserves_publication(
    failure_call: int,
) -> None:
    sources = ExpressionSources()
    calls = 0
    defect = RuntimeError("validator defect")

    def validate(expression: str) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise defect
        validate_scalar_expr(expression)

    resource = CfgResource(
        defaults,
        resolution=lambda: replace(sources.read(), validate_expression=validate),
        make_range=make_range,
    )
    before = resource.observe()
    with pytest.raises(RuntimeError) as caught:
        resource.edit(
            before.ref.revision, (CfgEdit(("a",), EvalValue("$device.value")),)
        )
    assert caught.value is defect
    assert resource.observe() == before


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("order", list(permutations((0, 1, 2))))
def test_whole_range_uses_new_geometry_before_step(centered: bool, order) -> None:
    spec = CenteredSweepSpec() if centered else SweepSpec()
    value = CenteredSweepValue(1.0, 2.0, 3) if centered else SweepValue(0.0, 2.0, 3)
    pairs = (
        [("center", 10.0), ("span", 8.0), ("step", 2.0)]
        if centered
        else [("start", 6.0), ("stop", 14.0), ("step", 2.0)]
    )
    resource = CfgResource(
        lambda: CfgSchema(
            CfgSectionSpec(fields={"range": spec}), CfgSectionValue({"range": value})
        ),
        resolution=Sources().read,
        make_range=make_range,
    )
    changed = resource.edit(
        resource.observe().ref.revision,
        (CfgEdit(("range",), dict(pairs[index] for index in order)),),
    )
    assert changed.status is CfgStatus.VALID
    accepted = resource.accept(changed.ref.revision)
    assert accepted.values["range"] == {"start": 6.0, "stop": 14.0, "expts": 5}
    observed = changed.tree.children["range"].value
    assert isinstance(observed, (SweepValue, CenteredSweepValue))
    assert observed.step == 2.0


@pytest.mark.parametrize(
    "payload", [{"expts": 5, "step": 2.0}, {"center": 8.0}, {"unknown": 1.0}]
)
def test_whole_range_rejection_is_atomic(payload) -> None:
    sources = Sources()
    schema = defaults()
    schema.spec.fields["range"] = CenteredSweepSpec(locked_center=1.0)
    schema.value.fields["range"] = CenteredSweepValue(1.0, 2.0, 3)
    resource = CfgResource(
        lambda: schema, resolution=sources.read, make_range=make_range
    )
    before = resource.observe()
    with pytest.raises(CfgInputError):
        resource.edit(
            before.ref.revision, (CfgEdit(("a",), 9.0), CfgEdit(("range",), payload))
        )
    assert resource.observe() == before
