"""Names and numeric fields share one fixed alias capture."""

from pathlib import Path

import pytest
from zcu_tools.gui.session.aliases import AliasRow, AliasTable
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
)
from zcu_tools.gui.session.expression import (
    ExpressionError,
    evaluate_numeric_expr,
    evaluate_scalar_expr,
)
from zcu_tools.gui.session.types import ParameterContext, RoleBinding
from zcu_tools.resources.entry import ComponentSchema, RoleResolutionError

from tests.gui.app.measure._cfg_fakes import parameter_capture


class LinkedComponent(ComponentSchema):
    rate: float = 4.0
    device: str = "yoko1"


def test_name_and_softlink_resolve_without_component_or_device_catalog() -> None:
    base = parameter_capture({"Q1": LinkedComponent(kind="alias/test")})
    rows = (AliasRow("qubit", "Q1"), AliasRow("flux_dev", "$qubit.device"))
    captured = base.bind_aliases(rows, {})
    assert captured.value("$flux_dev") == "yoko1"
    assert evaluate_scalar_expr("$flux_dev", captured) == "yoko1"
    assert evaluate_scalar_expr("$qubit.rate * 2", captured) == 8.0
    assert AliasTable(rows).names(base.document(), {}) == {
        "qubit": "Q1",
        "flux_dev": "yoko1",
    }
    assert captured.resolved_aliases()[1].resolved == "yoko1"
    with pytest.raises(ExpressionError, match="numeric|number"):
        evaluate_numeric_expr("$flux_dev", captured)
    for expression in ("$flux_dev + 1", "$flux_dev * 2", "str($flux_dev)"):
        with pytest.raises(ExpressionError):
            evaluate_scalar_expr(expression, captured)


@pytest.mark.parametrize("rows", [(), (AliasRow("flux_dev", ""),)])
def test_missing_or_unbound_name_reports_alias_without_concrete_fallback(
    rows: tuple[AliasRow, ...],
) -> None:
    captured = parameter_capture(
        {"flux_dev": LinkedComponent(kind="alias/test")}
    ).bind_aliases(rows, {})
    with pytest.raises(RoleResolutionError, match="flux_dev"):
        captured.value("$flux_dev")
    with pytest.raises(ExpressionError, match="flux_dev"):
        evaluate_scalar_expr("$flux_dev", captured)


@pytest.mark.parametrize(
    ("rows", "reason", "remedy"),
    [
        ((), "Unknown alias", "add"),
        ((AliasRow("flux_dev", ""),), "Alias is not bound", "bind"),
    ],
    ids=["missing", "unbound"],
)
def test_missing_or_unbound_name_message_names_alias_and_aliases_remedy(
    rows: tuple[AliasRow, ...], reason: str, remedy: str
) -> None:
    captured = parameter_capture().bind_aliases(rows, {})
    with pytest.raises(RoleResolutionError) as resolution:
        captured.value("$flux_dev")
    with pytest.raises(ExpressionError) as expression:
        evaluate_scalar_expr("$flux_dev", captured)
    # Consumers compare the short reason; only the user-facing text changes.
    assert resolution.value.reason == reason
    assert isinstance(expression.value.__cause__, RoleResolutionError)
    for message in (str(resolution.value), str(expression.value)):
        assert "$flux_dev" in message
        assert remedy in message and "Aliases" in message
        assert "None" not in message and "focus=" not in message


def test_changed_table_and_temporary_override_leave_old_name_capture_fixed() -> None:
    table = AliasTable((AliasRow("flux_dev", "yoko1"),))
    base = parameter_capture()
    old = base.bind_aliases(table.rows, {})
    assert evaluate_scalar_expr("$flux_dev", old) == "yoko1"
    temporary = base.bind_aliases(table.with_override("flux_dev", "yoko2").rows, {})
    assert evaluate_scalar_expr("$flux_dev", temporary) == "yoko2"
    assert table.rows == (AliasRow("flux_dev", "yoko1"),)
    table.replace((AliasRow("flux_dev", "yoko3"),))
    assert (
        evaluate_scalar_expr("$flux_dev", base.bind_aliases(table.rows, {})) == "yoko3"
    )
    assert evaluate_scalar_expr("$flux_dev", old) == "yoko1"


def test_historical_names_remain_readable_without_point_membership() -> None:
    base = parameter_capture()
    historical = ParameterContext(
        point=base.document(),
        library=base.library,
        target=base.target,
        roles=RoleBinding(None, {}, {"flux_dev": "yoko1"}),
    )
    assert historical.value("$flux_dev") == "yoko1"
    assert evaluate_scalar_expr("$flux_dev", historical) == "yoko1"
    with pytest.raises((AttributeError, ValueError)):
        historical.value("$flux_dev.value")


def test_unavailable_name_capture_propagates_source_fault_instead_of_old_names() -> (
    None
):
    base = parameter_capture()
    fault = ContextSourceFault(
        "point", Path("point.yaml"), "OSError", "Unreadable point"
    )
    captured = ParameterContext(
        point=base.document(),
        library=base.library,
        target=base.target,
        roles=RoleBinding(
            None, {}, {"flux_dev": "yoko1"}, aliases=(AliasRow("flux_dev", "yoko1"),)
        ),
        source_faults=(fault,),
    )
    with pytest.raises(ContextSourceUnavailableError):
        evaluate_scalar_expr("$flux_dev", captured)
