"""Measure-app adapters for shared finished-cfg lowering ports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.cfg import (
    CfgSchema,
    ExpressionResolver,
    ReferenceResolver,
    lower_finished_cfg,
    lower_resolved_cfg,
    validate_finished_cfg,
    validate_reference_kinds,
)

if TYPE_CHECKING:
    from zcu_tools.gui.cfg.lowering import LoweredData
    from zcu_tools.gui.session.types import ParameterContext

_REFERENCE_KINDS = frozenset({"module", "waveform"})


def _make_expression_resolver(
    parameters: ParameterContext | None,
) -> ExpressionResolver:
    from zcu_tools.gui.session.expression import evaluate_scalar_expr

    def resolve_expression(expr: str, /) -> int | float | complex:
        return evaluate_scalar_expr(expr, parameters)

    return resolve_expression


def _make_reference_resolver(parameters: ParameterContext) -> ReferenceResolver:
    from zcu_tools.gui.app.measure.cfg_schemas import (
        module_cfg_to_value,
        waveform_cfg_to_value,
    )

    def resolve_reference(kind: str, key: str, /) -> str | None:
        root = parameters.library_key(key)
        library = parameters.library
        if kind == "module":
            if root not in library.module_keys():
                return None
            spec, _ = module_cfg_to_value(library.get_module(root))
        elif kind == "waveform":
            if root not in library.waveform_keys():
                return None
            spec, _ = waveform_cfg_to_value(library.get_waveform(root))
        else:
            raise RuntimeError(f"Unsupported reference kind {kind!r}")
        return spec.label

    return resolve_reference


def validate_schema(schema: CfgSchema, parameters: ParameterContext | None) -> None:
    """Validate against a fixed role-aware capture; None has no references."""
    validate_reference_kinds(schema, _REFERENCE_KINDS)
    validate_finished_cfg(
        schema,
        resolve_reference=None
        if parameters is None
        else _make_reference_resolver(parameters),
    )


def schema_to_resolved_dict(schema: CfgSchema) -> dict[str, LoweredData]:
    """Freeze displayed values as pure data, without live resolvers."""
    validate_reference_kinds(schema, _REFERENCE_KINDS)
    return lower_resolved_cfg(schema)


def schema_to_raw_dict(
    schema: CfgSchema,
    parameters: ParameterContext | None,
) -> dict[str, LoweredData]:
    """Lower using fixed captured roles and pure-data sweep ranges.

    None allows pure calculation but parameter/library expressions fail.
    Finished-value, reference and source errors propagate without writes.
    """
    validate_reference_kinds(schema, _REFERENCE_KINDS)
    return lower_finished_cfg(
        schema,
        resolve_expression=_make_expression_resolver(parameters),
        resolve_reference=None
        if parameters is None
        else _make_reference_resolver(parameters),
    )
