"""Autoflux-local adapters for shared finished-cfg lowering ports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from zcu_tools.gui.cfg import (
    CfgSchema,
    ExpressionResolver,
    ReferenceResolver,
    lower_finished_cfg,
)
from zcu_tools.program.v2 import SweepCfg

from .module_adapter import module_cfg_shape_label, waveform_cfg_shape_label

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_expression_resolver(md: MetaDict) -> ExpressionResolver:
    from zcu_tools.gui.session.expression import evaluate_numeric_expr

    def resolve_expression(expr: str, /) -> int | float:
        return evaluate_numeric_expr(expr, md)

    return resolve_expression


def _make_reference_resolver(ml: ModuleLibrary) -> ReferenceResolver:
    def resolve_reference(
        kind: Literal["module", "waveform"], key: str, /
    ) -> str | None:
        if kind == "module":
            if key not in ml.modules:
                return None
            return module_cfg_shape_label(ml.modules[key])
        if key not in ml.waveforms:
            return None
        return waveform_cfg_shape_label(ml.waveforms[key])

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
    return lower_finished_cfg(
        schema,
        resolve_expression=None if md is None else _make_expression_resolver(md),
        resolve_reference=None if ml is None else _make_reference_resolver(ml),
        make_range=_make_sweep_range,
    )
