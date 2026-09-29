from __future__ import annotations

import pytest
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSchema,
    CfgSectionSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ScalarSpec,
    SweepSpec,
    make_default_value,
)
from zcu_tools.gui.cfg.binding import (
    AgentSweepKind,
    AgentSweepTarget,
    CfgDraft,
    LegacySettablePathError,
    SettablePathError,
    SettableTargetKind,
    SweepField,
)

from ._fakes import BindingPorts


def _draft(spec: CfgSectionSpec) -> CfgDraft:
    ports = BindingPorts()
    ports.options["devices"] = ()
    ports.options["arb_waveforms"] = ()
    return CfgDraft(
        CfgSchema(spec, make_default_value(spec)),
        evaluate_expression=ports.evaluate,
        provide_options=ports.provide,
        references=ports,
    )


def _mixed_draft() -> CfgDraft:
    variant = CfgSectionSpec(
        label="Variant",
        fields={
            "gain": ScalarSpec("Gain", float),
            "nested": CfgSectionSpec(fields={"name": ScalarSpec("Name", str)}),
        },
    )
    return _draft(
        CfgSectionSpec(
            fields={
                "count": ScalarSpec("Count", int),
                "mode": ScalarSpec("Mode", str, choices=["a", "b"]),
                "sweep": SweepSpec(),
                "centered": CenteredSweepSpec(),
                "module": ReferenceSpec(kind="module", allowed=[variant]),
                "literal": LiteralSpec("fixed"),
            }
        )
    )


def test_iterator_order_and_resolve_acceptance_are_identical() -> None:
    draft = _mixed_draft()
    targets = tuple(draft.iter_settable_targets())
    assert [target.path for target in targets] == [
        "count",
        "mode",
        "sweep.start",
        "sweep.stop",
        "sweep.expts",
        "sweep.step",
        "centered.center",
        "centered.span",
        "centered.expts",
        "centered.step",
        "module.ref",
        "module.gain",
        "module.nested.name",
    ]
    for target in targets:
        resolved = draft.resolve_target(target.path)
        assert (resolved.path, resolved.kind, resolved.value_type) == (
            target.path,
            target.kind,
            target.value_type,
        )
        assert resolved.get_value() == target.get_value()
        assert resolved.choices() == target.choices()
        assert resolved.affects_path_shape is target.affects_path_shape
    assert draft.resolve_target("module.ref").affects_path_shape is True
    assert draft.resolve_target("count").affects_path_shape is False


def test_scalar_exact_types_and_eval_value() -> None:
    draft = _mixed_draft()
    draft.set_target("count", 3)
    assert draft.resolve_target("count").get_value() == DirectValue(3)
    draft.set_target("count", EvalValue("unknown"))
    assert isinstance(draft.resolve_target("count").get_value(), EvalValue)
    with pytest.raises(SettablePathError, match="expects int"):
        draft.set_target("count", True)


def test_sweep_edges_use_canonical_rules() -> None:
    draft = _mixed_draft()
    draft.set_target("sweep.start", 2)
    draft.set_target("sweep.stop", 8.0)
    draft.set_target("sweep.expts", 4)
    assert draft.resolve_target("sweep.step").get_value() == pytest.approx(2.0)
    with pytest.raises(SettablePathError, match="integer"):
        draft.set_target("sweep.expts", 2.0)


def test_agent_whole_sweep_is_normalized_without_changing_gui_leaf_grammar() -> None:
    draft = _mixed_draft()
    changes: list[object] = []
    draft.on_change.connect(lambda: changes.append(draft.snapshot().value))
    target = draft.resolve_agent_target("sweep")
    assert isinstance(target, AgentSweepTarget)
    assert target.path == "sweep" and target.kind is AgentSweepKind.SWEEP
    actual = target.set_value({"start": 2.0, "stop": 8.0, "step": 2.2})
    assert actual.expts == 4
    assert actual.step == pytest.approx(2.0)
    assert len(changes) == 1
    assert draft.resolve_target("sweep.step").get_value() == pytest.approx(2.0)
    assert "sweep" not in [item.path for item in draft.iter_settable_targets()]
    before = draft.snapshot().value
    with pytest.raises(SettablePathError, match="conflict"):
        target.set_value({"start": 2.0, "stop": 8.0, "step": 1.0, "expts": 7})
    assert draft.snapshot().value == before
    with pytest.raises(SettablePathError, match="whole sweep"):
        draft.resolve_agent_target("sweep.start")
    draft.close()


def test_agent_expression_endpoint_and_step_use_resolved_bounds_before_commit() -> None:
    ports = BindingPorts()
    ports.expressions["md_x"] = 2.0
    spec = CfgSectionSpec(fields={"sweep": SweepSpec()})
    draft = CfgDraft(
        CfgSchema(spec, make_default_value(spec)),
        evaluate_expression=ports.evaluate,
        provide_options=ports.provide,
        references=ports,
    )
    sweep = draft.root.fields["sweep"]
    assert isinstance(sweep, SweepField)
    actual = sweep.set_agent_value(
        {"start": EvalValue("md_x"), "stop": 8.0, "step": 2.0}
    )
    assert isinstance(actual.start, EvalValue)
    assert actual.start.resolved == 2.0
    assert actual.expts == 4 and actual.step == pytest.approx(2.0)
    assert draft.resolve_agent_target("sweep").get_value() == actual
    before = draft.snapshot().value
    with pytest.raises((SettablePathError, TypeError, ValueError)):
        sweep.set_agent_value({"start": EvalValue("missing"), "stop": 8.0, "step": 2.0})
    assert draft.snapshot().value == before
    draft.close()


def test_agent_centered_sweep_accepts_locked_span_but_rejects_center() -> None:
    draft = _draft(
        CfgSectionSpec(fields={"centered": CenteredSweepSpec(locked_center=0.5)})
    )
    target = draft.resolve_agent_target("centered")
    assert isinstance(target, AgentSweepTarget)
    assert target.kind is AgentSweepKind.CENTERED_SWEEP
    actual = target.set_value({"span": 4.0, "step": 2.0})
    assert isinstance(actual, CenteredSweepValue)
    assert actual.center == 0.5
    assert actual.span == 4.0
    assert actual.expts == 3 and actual.step == pytest.approx(2.0)
    before = draft.snapshot().value
    with pytest.raises(SettablePathError, match="center is locked"):
        target.set_value({"center": 0.5, "span": 8.0, "expts": 5})
    assert draft.snapshot().value == before
    with pytest.raises(SettablePathError, match="missing expts or step"):
        target.set_value({"span": 4.0})
    assert draft.snapshot().value == before
    draft.close()


def test_sweep_target_observes_unfinished_text_and_typed_edit_recovers() -> None:
    draft = _mixed_draft()
    sweep = draft.root.fields["sweep"]
    assert isinstance(sweep, SweepField)
    sweep.set_text("expts", "1e")
    invalid = draft.resolve_target("sweep.expts").get_value()
    assert isinstance(invalid, DirectValue)
    assert invalid.raw == "1e" and invalid.value is None
    before = draft.snapshot()
    with pytest.raises(SettablePathError):
        draft.set_target("sweep.expts", 2.5)
    assert draft.snapshot().value == before.value
    draft.set_target("sweep.expts", 5)
    assert draft.resolve_target("sweep.expts").get_value() == 5
    assert sweep.is_valid()
    assert invalid.raw == "1e" and invalid.value is None
    draft.close()


def test_reference_bare_label_is_normalized_and_legacy_aliases_do_not_mutate() -> None:
    draft = _mixed_draft()
    draft.set_target("module.ref", "Variant")
    assert draft.resolve_target("module.ref").get_value() == "<Custom:Variant>"
    before = draft.snapshot().value
    with pytest.raises(LegacySettablePathError) as sweep_error:
        draft.set_target("sweep.sweep.start", 9.0)
    assert sweep_error.value.replacement == "sweep.start"
    with pytest.raises(LegacySettablePathError) as value_error:
        draft.set_target("module.value.gain", 0.5)
    assert value_error.value.replacement == "module.gain"
    assert draft.snapshot().value == before


def test_reference_catalog_provider_value_error_stays_unexpected() -> None:
    ports = BindingPorts()

    def fail_resolve(kind: str, key: str):
        del kind, key
        raise ValueError("provider corrupt")

    ports.resolve = fail_resolve  # type: ignore[method-assign]
    variant = CfgSectionSpec(
        label="Variant", fields={"gain": ScalarSpec("Gain", float)}
    )
    spec = CfgSectionSpec(
        fields={"module": ReferenceSpec(kind="module", allowed=[variant])}
    )
    draft = CfgDraft(
        CfgSchema(spec, make_default_value(spec)),
        evaluate_expression=ports.evaluate,
        provide_options=ports.provide,
        references=ports,
    )

    with pytest.raises(ValueError, match="provider corrupt"):
        draft.set_target("module.ref", "broken-library-entry")


@pytest.mark.parametrize("key", ("", "a.b", "$wire"))
def test_unrepresentable_field_key_fails_when_target_tree_is_built(key: str) -> None:
    draft = _draft(CfgSectionSpec(fields={key: ScalarSpec("X", int)}))
    with pytest.raises(SettablePathError, match="cannot be represented"):
        tuple(draft.iter_settable_targets())


@pytest.mark.parametrize("key", ("ref", "value"))
def test_reference_child_reserved_key_collision_fails(key: str) -> None:
    draft = _draft(
        CfgSectionSpec(
            fields={
                "module": ReferenceSpec(
                    kind="module",
                    allowed=[
                        CfgSectionSpec(
                            label="Variant", fields={key: ScalarSpec("X", int)}
                        )
                    ],
                )
            }
        )
    )
    with pytest.raises(SettablePathError, match="collides"):
        tuple(draft.iter_settable_targets())


def test_cached_target_fails_after_draft_close() -> None:
    draft = _mixed_draft()
    target = draft.resolve_target("count")
    draft.close()
    with pytest.raises(RuntimeError, match="closed"):
        target.get_value()
    with pytest.raises(RuntimeError, match="closed"):
        target.set_value(1)


def test_target_kinds_are_nominal() -> None:
    draft = _mixed_draft()
    kinds = {target.path: target.kind for target in draft.iter_settable_targets()}
    assert kinds["count"] is SettableTargetKind.SCALAR
    assert kinds["sweep.start"] is SettableTargetKind.SWEEP_EDGE
    assert kinds["module.ref"] is SettableTargetKind.REFERENCE_KEY


def test_all_production_measure_schemas_have_unambiguous_target_grammar() -> None:
    from zcu_tools.experiment.v2_gui.measure.registry import register_all
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.app.measure.registry import Registry
    from zcu_tools.resources.context import MetaDict, ModuleLibrary

    registry = Registry()
    register_all(registry)
    ctx = SessionEnv(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    ports = BindingPorts()
    ports.options["devices"] = ()
    ports.options["arb_waveforms"] = ()
    for name in registry.list_names():
        schema = registry.create(name).make_default_cfg(ctx)
        draft = CfgDraft(
            schema,
            evaluate_expression=ports.evaluate,
            provide_options=ports.provide,
            references=ports,
        )
        targets = tuple(draft.iter_settable_targets())
        assert len({target.path for target in targets}) == len(targets), name
