from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

logger = logging.getLogger(__name__)

from .types import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
)

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _resolve_eval(
    value: EvalValue,
    md: "Optional[MetaDict]",
    *,
    path: str,
    label: str,
    type_: type = float,
) -> Union[int, float]:
    """Resolve an EvalValue to a number, coerced to ``type_``.

    Prefers the snapshot ``value.resolved``; when absent (adapters may build an
    EvalValue without one), evaluates ``value.expr`` against ``md`` — lowering is
    the single place that owns resolution. Fails only if there is no snapshot and
    no md to evaluate against, or the expression itself is invalid.

    ``type_`` is the owning ScalarSpec's physical type; the result is coerced via
    ``coerce_eval_result`` (int spec → int, float spec → float) so that e.g. an
    ``EvalValue("ro_ch")`` for an int channel lowers to ``int`` rather than float.
    """
    from zcu_tools.gui.expression import coerce_eval_result

    if value.resolved is not None:
        resolved = value.resolved
        # The snapshot wins, but if md is available cross-check it against a
        # fresh evaluation: a mismatch means the snapshot is stale (md changed
        # after the field was set). Log it — never silently lower a stale value
        # without trace — but keep the snapshot to preserve existing semantics.
        if md is not None:
            from zcu_tools.gui.expression import evaluate_numeric_expr

            try:
                fresh = evaluate_numeric_expr(value.expr, md)
            except Exception:
                fresh = None
            # Compare coerced-to-spec-type so an int/float representation
            # difference (5 vs 5.0) is not flagged as a drift.
            if fresh is not None and isinstance(fresh, (int, float)):
                if coerce_eval_result(fresh, type_) != coerce_eval_result(
                    resolved, type_
                ):
                    logger.warning(
                        "Config field '%s' (%s): EvalValue %r snapshot %r differs "
                        "from current md evaluation %r; using snapshot",
                        path,
                        label,
                        value.expr,
                        resolved,
                        fresh,
                    )
    elif md is not None:
        from zcu_tools.gui.expression import evaluate_numeric_expr

        try:
            resolved = evaluate_numeric_expr(value.expr, md)
        except Exception as exc:
            raise RuntimeError(
                f"Config field '{path}' ({label}) expression {value.expr!r} "
                f"failed to evaluate: {exc}"
            ) from exc
    else:
        raise RuntimeError(
            f"Config field '{path}' ({label}) expression {value.expr!r} is unresolved"
        )
    if not isinstance(resolved, (int, float)):
        raise RuntimeError(
            f"Config field '{path}' ({label}) resolved to non-numeric value"
        )
    return coerce_eval_result(resolved, type_)


def _resolve_sweep_edge(
    value: object, md: "Optional[MetaDict]", *, path: str, label: str
) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, EvalValue):
        # Sweep edges have no per-edge ScalarSpec; scan values are always float.
        return float(_resolve_eval(value, md, path=path, label=label, type_=float))
    raise RuntimeError(f"Config field '{path}' ({label}) must be numeric")


def _section_to_dict_inner(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    ml: "Optional[ModuleLibrary]",
    path: list[str],
    md: "Optional[MetaDict]" = None,
) -> dict:
    result: dict[str, Any] = {}
    extra_keys = set(value.fields.keys()) - set(spec.fields.keys())
    if extra_keys:
        section = ".".join(path) or "<root>"
        extras = ", ".join(sorted(extra_keys))
        raise RuntimeError(f"Config section '{section}' has unknown fields: {extras}")
    for key, node_spec in spec.fields.items():
        node_val = value.fields.get(key)
        if node_val is None:
            if isinstance(node_spec, LiteralSpec):
                result[key] = node_spec.value
                continue
            if (
                isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec))
                and node_spec.optional
            ):
                continue  # disabled optional ModuleRef/WaveformRef → omit from output
            label = getattr(node_spec, "label", "") or key
            full_path = ".".join([*path, key])
            raise RuntimeError(f"Config field '{full_path}' ({label}) is missing")

        if isinstance(node_spec, ScalarSpec):
            assert isinstance(node_val, (DirectValue, EvalValue))
            if isinstance(node_val, DirectValue):
                if node_val.is_unset:
                    label = node_spec.label or key
                    full_path = ".".join([*path, key])
                    raise RuntimeError(f"Config field '{full_path}' ({label}) is unset")
                result[key] = node_val.value
            else:
                label = node_spec.label or key
                full_path = ".".join([*path, key])
                result[key] = _resolve_eval(
                    node_val, md, path=full_path, label=label, type_=node_spec.type
                )

        elif isinstance(node_spec, LiteralSpec):
            result[key] = node_spec.value

        elif isinstance(node_spec, SweepSpec):
            assert isinstance(node_val, SweepValue)
            from zcu_tools.notebook.utils import make_sweep

            start = _resolve_sweep_edge(
                node_val.start,
                md,
                path=".".join([*path, key, "start"]),
                label="Sweep start",
            )
            stop = _resolve_sweep_edge(
                node_val.stop,
                md,
                path=".".join([*path, key, "stop"]),
                label="Sweep stop",
            )
            result[key] = make_sweep(start, stop, expts=node_val.expts)

        elif isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec)):
            assert isinstance(node_val, (ModuleRefValue, WaveformRefValue))
            if not isinstance(node_val.value, CfgSectionValue):
                label = node_spec.label or key
                full_path = ".".join([*path, key])
                raise RuntimeError(f"Config field '{full_path}' ({label}) is missing")
            result[key] = _section_to_dict_inner(
                _find_allowed_spec(node_spec, node_val, ml),
                node_val.value,
                ml,
                [*path, key],
                md,
            )

        elif isinstance(node_spec, DeviceRefSpec):
            assert isinstance(node_val, DirectValue)
            if (
                node_val.is_unset
                or not isinstance(node_val.value, str)
                or not node_val.value
            ):
                label = node_spec.label or key
                full_path = ".".join([*path, key])
                raise RuntimeError(f"Config field '{full_path}' ({label}) is unset")
            result[key] = node_val.value

        elif isinstance(node_spec, CfgSectionSpec):
            assert isinstance(node_val, CfgSectionValue)
            result[key] = _section_to_dict_inner(
                node_spec, node_val, ml, [*path, key], md
            )

        else:
            raise TypeError(f"Unknown CfgNodeSpec type: {type(node_spec)}")

    return result


def _section_to_dict(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    ml: "Optional[ModuleLibrary]",
    md: "Optional[MetaDict]" = None,
) -> dict:
    """Public entry point for lowering a section; path starts at root."""
    return _section_to_dict_inner(spec, value, ml, [], md)


def _find_allowed_spec(
    ref_spec: Union[ModuleRefSpec, WaveformRefSpec],
    ref_val: Union[ModuleRefValue, WaveformRefValue],
    ml: "Optional[ModuleLibrary]",
) -> CfgSectionSpec:
    """Return the CfgSectionSpec from allowed[] that matches chosen_key's label."""
    chosen = ref_val.chosen_key
    if chosen.startswith("<Custom:"):
        if not chosen.endswith(">"):
            raise RuntimeError(f"Invalid custom reference key: {chosen!r}")
        label = chosen[len("<Custom:") : -1]
        for spec in ref_spec.allowed:
            if spec.label == label:
                return spec
        allowed = ", ".join(spec.label for spec in ref_spec.allowed)
        raise RuntimeError(
            f"Unknown custom reference label {label!r}; allowed labels: {allowed}"
        )

    if ml is None:
        raise RuntimeError(
            f"Cannot resolve library reference {chosen!r} without ModuleLibrary"
        )

    from zcu_tools.gui.cfg_schemas import module_cfg_to_value, waveform_cfg_to_value

    if isinstance(ref_spec, ModuleRefSpec):
        if chosen not in ml.modules:
            raise RuntimeError(f"Unknown module reference: {chosen!r}")
        chosen_spec, _ = module_cfg_to_value(ml.modules[chosen])
    else:
        if chosen not in ml.waveforms:
            raise RuntimeError(f"Unknown waveform reference: {chosen!r}")
        chosen_spec, _ = waveform_cfg_to_value(ml.waveforms[chosen])

    for spec in ref_spec.allowed:
        if spec.label == chosen_spec.label:
            return spec
    allowed = ", ".join(spec.label for spec in ref_spec.allowed)
    raise RuntimeError(
        f"Library reference {chosen!r} resolved to unsupported spec "
        f"{chosen_spec.label!r}; allowed labels: {allowed}"
    )


def schema_to_dict(
    schema: CfgSchema,
    ml: "Optional[ModuleLibrary]",
    md: "Optional[MetaDict]" = None,
) -> dict:
    """Lower a CfgSchema using the same section lowerer as CfgSchema.to_raw_dict().

    ``md`` lets lowering resolve any EvalValue that was built without a snapshot
    ``resolved``; omit it only when every EvalValue is already resolved.
    """
    return _section_to_dict_inner(schema.spec, schema.value, ml, [], md)
