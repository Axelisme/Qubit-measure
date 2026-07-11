from __future__ import annotations

import pytest
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
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
    CfgDraft,
    LegacySettablePathError,
    SettablePathError,
    SettableTargetKind,
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
    from zcu_tools.experiment.v2_gui.registry import register_all
    from zcu_tools.gui.app.main.adapter import ExpContext
    from zcu_tools.gui.app.main.registry import Registry
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    registry = Registry()
    register_all(registry)
    ctx = ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
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
