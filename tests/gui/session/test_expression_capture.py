"""Scalar expressions evaluated against fixed parameter captures."""

from collections.abc import Generator

import pytest
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.expression import ExpressionError, evaluate_scalar_expr
from zcu_tools.gui.session.types import ParameterContext, RoleBinding
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import ComponentSchema, RoleSpec, component_registry
from zcu_tools.resources.entry.schema import PointDocument


class Driver(ComponentSchema):
    rate: float


@pytest.fixture
def parameters() -> Generator[ParameterContext]:
    before = component_registry.kinds()
    component_registry.register("lowering/driver", Driver)
    try:
        point = PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "2.0",
                "general": {"created_at": "2026-10-08T00:00:00Z"},
                "components": {"Q1": {"kind": "lowering/driver", "rate": 4300.0}},
            }
        )
        library = LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        )
        yield ParameterContext(
            point=point,
            library=library,
            target=SavedPoint(entry_id="entry", point="working"),
            roles=RoleBinding(
                None,
                {"qubit": RoleSpec("lowering/driver")},
                {},
                aliases=(AliasRow("qubit", "Q1"),),
            ),
        )
    finally:
        component_registry.unregister("lowering/driver")
        assert component_registry.kinds() == before


def test_role_path_expression_uses_fixed_capture(parameters: ParameterContext) -> None:
    assert evaluate_scalar_expr("$qubit.rate * 2", parameters) == 8600.0
    assert evaluate_scalar_expr("Q1.rate + 100", parameters) == 4400.0


def test_no_parameters_allows_math_but_not_parameter_paths() -> None:
    assert evaluate_scalar_expr("sqrt(9) + 2", None) == 5.0
    with pytest.raises(ExpressionError):
        evaluate_scalar_expr("$qubit.rate", None)
