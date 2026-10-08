"""Captured parameter paths and finished pure-data lowering contracts."""

from collections.abc import Generator

import pytest
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.expression import ExpressionError, evaluate_scalar_expr
from zcu_tools.gui.session.types import ParameterContext, RoleBinding, SessionEnv
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import ComponentSchema, RoleSpec, component_registry
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class Driver(ComponentSchema):
    rate: float


@pytest.fixture
def parameters() -> Generator[ParameterContext]:
    before = component_registry.kinds()
    component_registry.register("lowering/driver", Driver)
    try:
        point = PointDocument(
            format="zcu.parameter-container",
            format_version="1.0",
            general=PointGeneral(created_at="2026-10-08T00:00:00Z"),
            components={"Q1": {"kind": "lowering/driver", "rate": 4300.0}},
        )
        library = LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        )
        yield ParameterContext(
            point=point,
            library=library,
            target=EntryDraft(entry_id="entry"),
            roles=RoleBinding(
                "Q1", {"qubit": RoleSpec("lowering/driver")}, {"qubit": "Q1"}
            ),
        )
    finally:
        component_registry.unregister("lowering/driver")
        assert component_registry.kinds() == before


def test_role_path_expression_uses_fixed_capture(parameters: ParameterContext) -> None:
    assert evaluate_scalar_expr("qubit.rate * 2", parameters) == 8600.0
    assert evaluate_scalar_expr("Q1.rate + 100", parameters) == 4400.0


def test_no_parameters_allows_math_but_not_parameter_paths() -> None:
    assert evaluate_scalar_expr("sqrt(9) + 2", None) == 5.0
    with pytest.raises(ExpressionError):
        evaluate_scalar_expr("qubit.rate", None)


def test_lowering_uses_capture_and_returns_pure_data_range(
    parameters: ParameterContext,
) -> None:
    schema = CfgSchema(
        CfgSectionSpec(
            fields={"freq": ScalarSpec("Frequency", float), "sweep": SweepSpec()}
        ),
        CfgSectionValue(
            fields={"freq": EvalValue("qubit.rate"), "sweep": SweepValue(0.0, 1.0, 3)}
        ),
    )
    lowered = schema_to_raw_dict(schema, parameters)
    assert lowered == {
        "freq": 4300.0,
        "sweep": {"start": 0.0, "stop": 1.0, "expts": 3, "step": 0.5},
    }


@pytest.mark.parametrize("value", [(1, 2), {1: "bad"}, object()])
def test_finished_lowering_rejects_unsupported_literal(value: object) -> None:
    schema = CfgSchema(
        CfgSectionSpec(fields={"payload": LiteralSpec(value)}),
        CfgSectionValue(fields={"payload": DirectValue(value)}),
    )
    with pytest.raises(RuntimeError):
        schema_to_raw_dict(schema, None)


def test_finished_literal_data_is_independent() -> None:
    value = {"items": [1, 2]}
    schema = CfgSchema(
        CfgSectionSpec(fields={"payload": LiteralSpec(value)}),
        CfgSectionValue(fields={"payload": DirectValue(value)}),
    )
    lowered = schema_to_raw_dict(schema, None)
    value["items"].append(3)
    assert lowered == {"payload": {"items": [1, 2]}}


def test_path_seeds_use_roles_and_explicit_fallback(
    parameters: ParameterContext,
) -> None:
    from zcu_lab.v2._support.measure.seeds import path_seed, scaled_path_seed

    environment = SessionEnv(parameters=parameters, soc=None, soccfg=None)
    value = path_seed("qubit.rate", fallback=100.0).resolve(environment)
    assert isinstance(value, EvalValue) and value.expr == "qubit.rate"
    scaled = scaled_path_seed("qubit.rate", factor=2.0, fallback_value=100.0).resolve(
        environment
    )
    assert isinstance(scaled, EvalValue)
    assert evaluate_scalar_expr(scaled.expr, parameters) == 8600.0
    empty = SessionEnv(parameters=None, soc=None, soccfg=None)
    assert path_seed("qubit.rate", fallback=100.0).resolve(empty) == 100.0
