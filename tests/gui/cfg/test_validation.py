"""Characterization tests for finished-cfg validation and error order."""

from __future__ import annotations

import pytest
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    lower_finished_cfg,
    validate_finished_cfg,
)


def _schema(
    spec_fields: dict[str, CfgNodeSpec], value_fields: dict[str, CfgNodeValue | None]
) -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(fields=spec_fields),
        value=CfgSectionValue(fields=value_fields),
    )


def _missing_expression(expr: str) -> float:
    raise RuntimeError(f"Missing parameter: {expr!r}")


@pytest.mark.parametrize("kind", ["module", "waveform"])
def test_linked_empty_input_can_await_shape_but_cannot_lower(kind: str) -> None:
    schema = _schema(
        {
            "ref": ReferenceSpec(
                kind=kind,
                allowed=[
                    CfgSectionSpec(
                        label="Shape", fields={"count": ScalarSpec("Count", int)}
                    )
                ],
            )
        },
        {"ref": ReferenceValue("driver.module.pi", CfgSectionValue())},
    )

    # Static input admission must not read an unavailable reference source.
    def unavailable(kind: str, key: str, /) -> str | None:
        raise LookupError(f"Source unavailable: {kind}:{key}")

    validate_finished_cfg(schema, resolve_reference=unavailable)
    with pytest.raises(LookupError, match="Source unavailable"):
        lower_finished_cfg(
            schema, resolve_expression=None, resolve_reference=unavailable
        )
    with pytest.raises(RuntimeError, match="Unknown .* reference"):
        lower_finished_cfg(
            schema, resolve_expression=None, resolve_reference=lambda kind, key: None
        )
    with pytest.raises(RuntimeError, match="count.*missing"):
        lower_finished_cfg(
            schema, resolve_expression=None, resolve_reference=lambda kind, key: "Shape"
        )


@pytest.mark.parametrize(
    "value",
    [
        ReferenceValue("<Custom:Shape>", CfgSectionValue()),
        ReferenceValue(
            "driver.module.pi", CfgSectionValue(fields={"count": DirectValue(1.5)})
        ),
        ReferenceValue("<Custom:Shape", CfgSectionValue()),
        ReferenceValue("driver.module.pi", CfgSectionValue(), error="source fault"),
    ],
)
def test_static_reference_input_exemption_does_not_mask_faults(
    value: ReferenceValue,
) -> None:
    schema = _schema(
        {
            "ref": ReferenceSpec(
                kind="module",
                allowed=[
                    CfgSectionSpec(
                        label="Shape", fields={"count": ScalarSpec("Count", int)}
                    )
                ],
            )
        },
        {"ref": value},
    )
    with pytest.raises(RuntimeError):
        validate_finished_cfg(schema, resolve_reference=lambda kind, key: "Shape")


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        CfgSectionSpec(label="LiteralOnly", fields={"type": LiteralSpec("none")}),
        CfgSectionSpec(
            label="OptionalOnly",
            fields={
                "optional": ReferenceSpec(
                    kind="module",
                    allowed=[CfgSectionSpec(label="Zero", fields={})],
                    optional=True,
                )
            },
        ),
    ],
)
def test_finished_reference_requires_literal_and_optional_entries(
    shape: CfgSectionSpec,
    dynamic: bool,
) -> None:
    schema = _schema(
        {"ref": ReferenceSpec(kind="module", allowed=[shape])},
        {"ref": ReferenceValue("library.pending", CfgSectionValue())},
    )
    validate_finished_cfg(schema, resolve_reference=None)
    with pytest.raises(RuntimeError, match="ref\\.(type|optional).*missing"):
        lower_finished_cfg(
            schema,
            resolve_expression=(lambda _: 1.0) if dynamic else None,
            resolve_reference=lambda kind, key: shape.label,
        )


@pytest.mark.parametrize("dynamic", [False, True])
def test_finished_nested_reference_rejects_incomplete_payload(dynamic: bool) -> None:
    inner = CfgSectionSpec(label="Inner", fields={"type": LiteralSpec("none")})
    outer = CfgSectionSpec(
        label="Outer", fields={"nested": ReferenceSpec(kind="module", allowed=[inner])}
    )
    schema = _schema(
        {"ref": ReferenceSpec(kind="module", allowed=[outer])},
        {
            "ref": ReferenceValue(
                "outer",
                CfgSectionValue(
                    fields={"nested": ReferenceValue("inner", CfgSectionValue())}
                ),
            )
        },
    )
    validate_finished_cfg(schema, resolve_reference=lambda kind, key: "Outer")
    with pytest.raises(RuntimeError, match="ref.nested.type.*missing"):
        lower_finished_cfg(
            schema,
            resolve_expression=(lambda _: 1.0) if dynamic else None,
            resolve_reference=lambda kind, key: "Outer" if key == "outer" else "Inner",
        )


def test_finished_reference_allows_complete_disabled_optional_payload() -> None:
    shape = CfgSectionSpec(
        label="OptionalOnly",
        fields={
            "optional": ReferenceSpec(
                kind="module",
                allowed=[CfgSectionSpec(label="Zero", fields={})],
                optional=True,
            )
        },
    )
    schema = _schema(
        {"ref": ReferenceSpec(kind="module", allowed=[shape])},
        {
            "ref": ReferenceValue(
                "library.complete", CfgSectionValue(fields={"optional": None})
            )
        },
    )
    assert lower_finished_cfg(
        schema,
        resolve_expression=None,
        resolve_reference=lambda kind, key: "OptionalOnly",
    ) == {"ref": {}}


def test_complex_live_value_cannot_lower_to_real_scalar() -> None:
    schema = _schema(
        {"frequency": ScalarSpec("Frequency", float)},
        {"frequency": EvalValue("frequency", resolved=5.0)},
    )
    with pytest.raises(RuntimeError, match="cannot target a real field"):
        lower_finished_cfg(
            schema, resolve_expression=lambda _: 1 + 2j, resolve_reference=None
        )


def test_static_validation_fast_fails_missing_and_extra_fields() -> None:
    missing = _schema({"count": ScalarSpec("Count", int)}, {})
    with pytest.raises(RuntimeError) as missing_error:
        validate_finished_cfg(missing, resolve_reference=None)
    assert str(missing_error.value) == (
        "Config field 'count' is missing from the value"
    )

    extra = _schema(
        {"count": ScalarSpec("Count", int)},
        {"count": DirectValue(1), "extra": DirectValue(2)},
    )
    with pytest.raises(RuntimeError) as extra_error:
        validate_finished_cfg(extra, resolve_reference=None)
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
            "got DirectValue(value='changed', raw=None, error=None)",
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
    spec: CfgNodeSpec, value: CfgNodeValue, message: str
) -> None:
    schema = _schema({"value": spec}, {"value": value})

    with pytest.raises(RuntimeError) as exc_info:
        validate_finished_cfg(schema, resolve_reference=None)

    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ("spec", "value", "message"),
    [
        (
            SweepSpec("Range"),
            DirectValue(1.0),
            "Config field 'value' must be a SweepValue, got DirectValue",
        ),
        (
            CenteredSweepSpec("Range"),
            DirectValue(1.0),
            "Config field 'value' must be a CenteredSweepValue, got DirectValue",
        ),
        (
            CfgSectionSpec(fields={}),
            DirectValue(1.0),
            "Config field 'value' must be a CfgSectionValue, got DirectValue",
        ),
        (
            ScalarSpec("Value", float),
            None,
            "Config field 'value' is None but is not a disabled optional ref",
        ),
    ],
)
def test_static_validation_rejects_wrong_carriers(
    spec: CfgNodeSpec, value: CfgNodeValue | None, message: str
) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        validate_finished_cfg(
            _schema({"value": spec}, {"value": value}), resolve_reference=None
        )
    assert str(exc_info.value) == message


def test_dynamic_nested_ranges_validate_in_field_order_before_lowering() -> None:
    calls: list[str] = []
    values = {"start": 1.0, "stop": 2.0, "center": 10.0}

    def resolve(expr: str, /) -> float:
        calls.append(expr)
        return values[expr]

    schema = _schema(
        {
            "nested": CfgSectionSpec(
                fields={
                    "linear": SweepSpec("Linear"),
                    "centered": CenteredSweepSpec("Centered", locked_center=10.0),
                }
            )
        },
        {
            "nested": CfgSectionValue(
                fields={
                    "linear": SweepValue(EvalValue("start"), EvalValue("stop"), 5),
                    "centered": CenteredSweepValue(
                        center=EvalValue("center"), span=4.0, expts=5
                    ),
                }
            )
        },
    )
    assert lower_finished_cfg(
        schema, resolve_expression=resolve, resolve_reference=None
    ) == {
        "nested": {
            "linear": {"start": 1.0, "stop": 2.0, "expts": 5, "step": 0.25},
            "centered": {"start": 8.0, "stop": 12.0, "expts": 5, "step": 1.0},
        }
    }
    # Dynamic validation visits authored edges first; lowering resolves them again.
    assert calls == ["start", "stop", "center", "center", "start", "stop", "center"]


def test_dynamic_locked_center_rejects_resolved_mismatch() -> None:
    schema = _schema(
        {"range": CenteredSweepSpec("Range", locked_center=10.0)},
        {"range": CenteredSweepValue(center=EvalValue("center"), span=4.0, expts=5)},
    )
    with pytest.raises(RuntimeError) as exc_info:
        lower_finished_cfg(
            schema, resolve_expression=lambda _: 11.0, resolve_reference=None
        )
    assert str(exc_info.value) == (
        "Config field 'range.center' (Sweep center) is locked to 10.0, got 11.0"
    )


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
        lower_finished_cfg(
            schema, resolve_expression=_missing_expression, resolve_reference=None
        )

    assert str(exc_info.value) == (
        "Config field 'literal' is a locked literal (must be 'fixed'), "
        "got DirectValue(value='changed', raw=None, error=None)"
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
        lower_finished_cfg(
            schema, resolve_expression=None, resolve_reference=lambda kind, key: None
        )

    assert str(exc_info.value) == (
        "Config field 'literal' is a locked literal (must be 'fixed'), "
        "got DirectValue(value='changed', raw=None, error=None)"
    )


def test_dynamic_validation_precedes_lowering_error() -> None:
    schema = _schema(
        {"count": ScalarSpec("Count", int)},
        {"count": DirectValue(None)},
    )

    with pytest.raises(RuntimeError) as exc_info:
        lower_finished_cfg(
            schema, resolve_expression=_missing_expression, resolve_reference=None
        )

    assert str(exc_info.value) == (
        "Config field 'count' (Count) is unset (no value to lower)"
    )


def test_dynamic_expression_error_uses_exact_path_and_text() -> None:
    schema = _schema(
        {"frequency": ScalarSpec("Frequency", float)},
        {"frequency": EvalValue("missing")},
    )

    with pytest.raises(RuntimeError) as exc_info:
        lower_finished_cfg(
            schema, resolve_expression=_missing_expression, resolve_reference=None
        )

    assert str(exc_info.value) == (
        "Config field 'frequency' (Frequency) expression 'missing' failed: "
        "Missing parameter: 'missing'"
    )
