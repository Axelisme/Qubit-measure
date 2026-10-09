"""Expression namespaces use fixed aliases and never legacy bare-role shorthand."""

import pytest
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.expression import ExpressionError, evaluate_scalar_expr
from zcu_tools.gui.session.types import ParameterContext, RoleBinding
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import ComponentSchema, RoleSpec
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class NumericComponent(ComponentSchema):
    rate: float


def _parameters(*, legacy: bool = False) -> ParameterContext:
    point = PointDocument.model_construct(
        format="zcu.parameter-container",
        format_version="1.0",
        general=PointGeneral(created_at="2026-10-09T00:00:00Z", ext={"offset": 2.0}),
        components={
            "Q1": NumericComponent(kind="expression/test", rate=4.0),
            "Q2": NumericComponent(kind="expression/test", rate=5.0),
            "qubit": NumericComponent(kind="expression/test", rate=3.0),
        },
    )
    parameters = ParameterContext(
        point=point,
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        ),
        target=EntryDraft(entry_id="entry"),
        roles=RoleBinding(None, {"qubit": RoleSpec("expression/test")}, {"qubit": "Q1"})
        if legacy
        else None,
    )
    return (
        parameters
        if legacy
        else parameters.bind_aliases((AliasRow("qubit", "Q1"),), {})
    )


@pytest.mark.parametrize("legacy", [False, True], ids=["authored", "legacy-fixed-role"])
def test_dollar_alias_and_bare_component_remain_distinct(legacy: bool) -> None:
    parameters = _parameters(legacy=legacy)
    assert evaluate_scalar_expr("$qubit.rate ** 2 + qubit.rate", parameters) == 19.0
    assert evaluate_scalar_expr("general.ext.offset + Q2.rate", parameters) == 7.0


def test_alias_replacement_uses_new_capture_without_changing_old_evaluation() -> None:
    original = _parameters()
    replacement = original.bind_aliases((AliasRow("qubit", "Q2"),), {})
    assert evaluate_scalar_expr("$qubit.rate", original) == 4.0
    assert evaluate_scalar_expr("$qubit.rate", replacement) == 5.0
    assert evaluate_scalar_expr("$qubit.rate", original) == 4.0


def test_missing_alias_does_not_read_a_same_named_component() -> None:
    parameters = _parameters().bind_aliases((AliasRow("qubit", ""),), {})
    assert evaluate_scalar_expr("qubit.rate", parameters) == 3.0
    with pytest.raises(ExpressionError):
        evaluate_scalar_expr("$qubit.rate", parameters)


@pytest.mark.parametrize(
    "expression",
    ["$qubit", "$qubit..rate", "$ qubit.rate", "$qubit.rate[0]", "$qubit.rate()"],
)
def test_invalid_alias_expression_is_rejected(expression: str) -> None:
    with pytest.raises(ExpressionError):
        evaluate_scalar_expr(expression, _parameters())


def test_parameter_reader_fault_is_not_reported_as_invalid_expression(
    monkeypatch,
) -> None:
    parameters = _parameters()

    def broken_source(path: str) -> object:
        raise AttributeError(f"Source implementation failed for {path}")

    monkeypatch.setattr(parameters, "lookup", broken_source)
    with pytest.raises(AttributeError, match="Source implementation failed"):
        evaluate_scalar_expr("$qubit.rate", parameters)


def test_context_free_arithmetic_does_not_grant_parameter_access() -> None:
    assert evaluate_scalar_expr("2 ** 3 + 4", None) == 12
    with pytest.raises(ExpressionError, match="context"):
        evaluate_scalar_expr("$qubit.rate", None)
