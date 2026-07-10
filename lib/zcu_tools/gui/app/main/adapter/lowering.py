"""Measure-app adapters for shared finished-cfg lowering ports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.cfg import (
    CfgSchema,
    ExpressionResolver,
    ReferenceResolver,
    lower_finished_cfg,
    validate_finished_cfg,
    validate_reference_kinds,
)
from zcu_tools.program.v2 import SweepCfg

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

_REFERENCE_KINDS = frozenset({"module", "waveform"})


def _make_expression_resolver(md: MetaDict) -> ExpressionResolver:
    from zcu_tools.gui.session.expression import evaluate_numeric_expr

    def resolve_expression(expr: str, /) -> int | float:
        return evaluate_numeric_expr(expr, md)

    return resolve_expression


def _make_reference_resolver(ml: ModuleLibrary) -> ReferenceResolver:
    from zcu_tools.gui.app.main.cfg_schemas import (
        module_cfg_to_value,
        waveform_cfg_to_value,
    )

    def resolve_reference(kind: str, key: str, /) -> str | None:
        if kind == "module":
            if key not in ml.modules:
                return None
            spec, _ = module_cfg_to_value(ml.modules[key])
            return spec.label
        if kind == "waveform":
            if key not in ml.waveforms:
                return None
            spec, _ = waveform_cfg_to_value(ml.waveforms[key])
            return spec.label
        raise RuntimeError(f"Unsupported reference kind {kind!r}")

    return resolve_reference


def _make_sweep_range(start: float, stop: float, /, *, expts: int) -> SweepCfg:
    if expts == 1:
        assert stop == start, (
            f"for expts == 1, stop must equal start, got start={start}, stop={stop}"
        )
        step = 0.0
    else:
        step = (stop - start) / (expts - 1)

    normalized_stop = start + step * (expts - 1)

    assert expts > 0, f"expts must be greater than 0, but got {expts}"
    if expts == 1:
        assert step == 0, f"for expts == 1, step must be 0, but got {step}"
    else:
        assert step != 0, f"step must not be zero when expts > 1, but got {step}"

    return SweepCfg(
        start=start,
        stop=normalized_stop,
        expts=expts,
        step=step,
    )


def validate_schema(schema: CfgSchema, ml: ModuleLibrary | None) -> None:
    """Validate the static contract using the current measure library."""
    validate_reference_kinds(schema, _REFERENCE_KINDS)
    validate_finished_cfg(
        schema,
        resolve_reference=None if ml is None else _make_reference_resolver(ml),
    )


def schema_to_raw_dict(
    schema: CfgSchema,
    md: MetaDict | None,
    ml: ModuleLibrary | None,
) -> dict[str, object]:
    """Lower through the shared algorithm with measure-owned runtime policy."""
    validate_reference_kinds(schema, _REFERENCE_KINDS)
    return lower_finished_cfg(
        schema,
        resolve_expression=None if md is None else _make_expression_resolver(md),
        resolve_reference=None if ml is None else _make_reference_resolver(ml),
        make_range=_make_sweep_range,
    )
