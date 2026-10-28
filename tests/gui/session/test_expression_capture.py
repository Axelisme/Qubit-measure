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
                "schema_version": 1,
                "general": {"created_at": "2026-10-08T00:00:00Z"},
                "components": {"Q1": {"kind": "lowering/driver", "rate": 4300.0}},
            }
        )
        library = LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
        )
        yield ParameterContext(
            point=point,
            library=library,
            target=SavedPoint(entry_name="entry", point="working"),
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


@pytest.mark.parametrize("target_type", [int, float])
@pytest.mark.parametrize("bounds", [(4400.0, 4500.0), (4300.0, 4400.0)])
def test_eval_ref_resolves_captured_paths_and_inclusive_bounds(
    parameters: ParameterContext, target_type: type, bounds: tuple[float, float]
) -> None:
    from zcu_tools.gui.session import expression

    ref = expression.EvalRef("Q1.rate + 100", target_type, *bounds)
    result = expression.resolve_eval_ref(ref, parameters)
    assert result == 4400
    assert type(result) is target_type


@pytest.mark.parametrize(
    ("expr", "target_type", "message"),
    [
        ("-1", float, "value -1.0 out of range"),
        ("11", int, "value 11 out of range"),
        ("1.5", int, "not an integer"),
        ("Q1.rate", float, "requires a context"),
        ("sqrt(-1)", float, "math domain error"),
    ],
)
def test_eval_ref_rejects_invalid_or_out_of_range_values(
    expr: str, target_type: type, message: str
) -> None:
    from zcu_tools.gui.session import expression

    with pytest.raises(ExpressionError, match=message):
        expression.resolve_eval_ref(expression.EvalRef(expr, target_type, 0, 10), None)


def test_eval_ref_source_failure_propagates_but_trial_returns_diagnostic(
    parameters: ParameterContext, caplog: pytest.LogCaptureFixture
) -> None:
    from zcu_tools.gui.session import expression
    from zcu_tools.gui.session.predictor_control import PredictorControlFacet
    from zcu_tools.gui.session.predictor_trial import PredictorTrial
    from zcu_tools.gui.session.services.predictor import PredictorService

    failure = OSError("alias source offline")
    capture = parameters.bind_aliases(
        (AliasRow("qubit", "Q1"),), {}, alias_error=failure
    )
    ref = expression.EvalRef("$qubit.rate", float, 0, 10000)
    with pytest.raises(OSError, match="alias source offline") as raised:
        expression.resolve_eval_ref(ref, capture)
    assert raised.value is failure
    trial = PredictorTrial(PredictorControlFacet(PredictorService()))
    resolved = trial.resolve_calibration_frequency_mhz(ref, lambda: capture)
    assert resolved.frequency_mhz is None
    assert resolved.error == "alias source offline"
    assert not caplog.records


def test_no_parameters_allows_math_but_not_parameter_paths() -> None:
    assert evaluate_scalar_expr("sqrt(9) + 2", None) == 5.0
    with pytest.raises(ExpressionError):
        evaluate_scalar_expr("$qubit.rate", None)
