"""Autoflux-local adapters for shared finished-cfg lowering ports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.cfg import (
    CfgSchema,
    ExpressionResolver,
    ReferenceResolver,
    lower_finished_cfg,
    validate_reference_kinds,
)
from zcu_tools.gui.measure_cfg import program_shape_for_input
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
    resolved: dict[tuple[str, str], str | None] = {}

    def resolve_reference(kind: str, key: str, /) -> str | None:
        cache_key = (kind, key)
        if cache_key in resolved:
            return resolved[cache_key]
        if kind == "module":
            if key not in ml.modules:
                label = None
            else:
                label = program_shape_for_input("module", ml.modules[key]).label
        elif kind == "waveform":
            if key not in ml.waveforms:
                label = None
            else:
                label = program_shape_for_input("waveform", ml.waveforms[key]).label
        else:
            raise RuntimeError(f"Unsupported reference kind {kind!r}")
        resolved[cache_key] = label
        return label

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


def schema_to_raw_dict(
    schema: CfgSchema,
    md: MetaDict | None,
    ml: ModuleLibrary | None,
) -> dict[str, object]:
    validate_reference_kinds(schema, _REFERENCE_KINDS)
    return lower_finished_cfg(
        schema,
        resolve_expression=None if md is None else _make_expression_resolver(md),
        resolve_reference=None if ml is None else _make_reference_resolver(ml),
        make_range=_make_sweep_range,
    )
