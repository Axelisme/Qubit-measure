"""Characterization tests for finished-cfg validation and error order."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter.lowering import (
    schema_to_raw_dict,
    validate_schema,
)
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.meta_tool import MetaDict


def _schema(
    spec_fields: dict[str, object], value_fields: dict[str, object]
) -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(fields=spec_fields),  # type: ignore[arg-type]
        value=CfgSectionValue(fields=value_fields),  # type: ignore[arg-type]
    )


def _library(*, modules: dict[str, object] | None = None) -> MagicMock:
    ml = MagicMock()
    ml.modules = {} if modules is None else modules
    ml.waveforms = {}
    return ml


def test_static_validation_fast_fails_missing_and_extra_fields() -> None:
    missing = _schema({"count": ScalarSpec("Count", int)}, {})
    with pytest.raises(RuntimeError) as missing_error:
        validate_schema(missing, None)
    assert str(missing_error.value) == (
        "Config field 'count' is missing from the value"
    )

    extra = _schema(
        {"count": ScalarSpec("Count", int)},
        {"count": DirectValue(1), "extra": DirectValue(2)},
    )
    with pytest.raises(RuntimeError) as extra_error:
        validate_schema(extra, None)
    assert str(extra_error.value) == (
        "Config section '<root>' has unknown fields: extra"
    )


@pytest.mark.parametrize(
    ("spec", "value", "message"),
    [
        (
            LiteralSpec("fixed"),
            DirectValue("changed"),
            "Config field 'value' is a locked literal (must be 'fixed'), "
            "got DirectValue(value='changed')",
        ),
        (
            ScalarSpec("Count", int),
            DirectValue(1.5),
            "Config field 'value' value 1.5 is not compatible with spec type int",
        ),
        (
            ScalarSpec("Count", int, choices=[1, 2]),
            DirectValue(3),
            "Config field 'value' value 3 is not in allowed choices [1, 2]",
        ),
    ],
)
def test_static_validation_uses_exact_literal_type_and_choice_errors(
    spec: object, value: object, message: str
) -> None:
    schema = _schema({"value": spec}, {"value": value})

    with pytest.raises(RuntimeError) as exc_info:
        validate_schema(schema, None)

    assert str(exc_info.value) == message


def test_static_error_precedes_expression_error() -> None:
    schema = _schema(
        {
            "literal": LiteralSpec("fixed"),
            "expression": ScalarSpec("Expression", float),
        },
        {
            "literal": DirectValue("changed"),
            "expression": EvalValue("missing"),
        },
    )

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, MetaDict(), None)

    assert str(exc_info.value) == (
        "Config field 'literal' is a locked literal (must be 'fixed'), "
        "got DirectValue(value='changed')"
    )


def test_static_error_precedes_reference_error() -> None:
    pulse = CfgSectionSpec(label="Pulse", fields={})
    schema = _schema(
        {
            "literal": LiteralSpec("fixed"),
            "module": ReferenceSpec(kind="module", allowed=[pulse]),
        },
        {
            "literal": DirectValue("changed"),
            "module": ReferenceValue("missing", CfgSectionValue()),
        },
    )

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, None, _library())

    assert str(exc_info.value) == (
        "Config field 'literal' is a locked literal (must be 'fixed'), "
        "got DirectValue(value='changed')"
    )


def test_dynamic_validation_precedes_lowering_error() -> None:
    schema = _schema(
        {"count": ScalarSpec("Count", int)},
        {"count": DirectValue(None)},
    )

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, MetaDict(), None)

    assert str(exc_info.value) == (
        "Config field 'count' (Count) is unset (no value to lower)"
    )


def test_dynamic_expression_error_uses_exact_path_and_text() -> None:
    schema = _schema(
        {"frequency": ScalarSpec("Frequency", float)},
        {"frequency": EvalValue("missing")},
    )

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, MetaDict(), None)

    assert str(exc_info.value) == (
        "Config field 'frequency' (Frequency) expression 'missing' failed: "
        "Variable 'missing' is not defined in MetaDict"
    )
