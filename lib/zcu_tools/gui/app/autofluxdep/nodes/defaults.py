"""Adapter-shaped default cfg schemas for autofluxdep nodes.

Fresh autofluxdep placements reuse the corresponding measure-gui adapter's spec
and copy its default value tree, then apply node-local runtime defaults before the
UI sees it. Autofluxdep adds its run-time ``generation`` section and a logical
projection for the node builder.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from zcu_tools.gui.app.autofluxdep.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    NodeCfgSchema,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.session.types import ExpContext
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

GENERATION_GROUP_LABELS: dict[str, str] = {
    "sweep": "Sweep generation",
    "timing": "Timing / relax",
    "feedback": "Feedback / adaptive",
    "fit": "Fit behavior",
    "safety": "Safety gates",
}


@dataclass(frozen=True)
class GenerationField:
    logical_key: str
    field_key: str
    spec: ScalarSpec | SweepSpec
    default: Any
    group_key: str
    group_label: str


def generation_field(
    logical_key: str,
    field_key: str,
    spec: ScalarSpec | SweepSpec,
    default: Any,
    *,
    group: str,
    group_label: str | None = None,
) -> GenerationField:
    if not group:
        raise ValueError("generation field group must be non-empty")
    return GenerationField(
        logical_key=logical_key,
        field_key=field_key,
        spec=spec,
        default=default,
        group_key=group,
        group_label=group_label or GENERATION_GROUP_LABELS.get(group, group),
    )


def adapter_node_schema(
    adapter_cls: type[Any],
    ctx: Any | None,
    *,
    logical_paths: Mapping[str, str],
    generation_fields: tuple[GenerationField, ...] = (),
    default_overrides: Mapping[str, Any] | None = None,
) -> NodeCfgSchema:
    """Build a ``NodeCfgSchema`` from a copied measure-gui adapter cfg shape."""
    schema = adapter_cls().make_default_cfg(_ensure_context(ctx))
    spec_fields = dict(schema.spec.fields)
    value_fields = deepcopy(schema.value.fields)
    projection = dict(logical_paths)

    if generation_fields:
        generation_spec_fields: dict[str, Any] = {}
        generation_value_fields: dict[str, Any] = {}
        for field in generation_fields:
            group_spec = generation_spec_fields.setdefault(
                field.group_key,
                CfgSectionSpec(label=field.group_label, fields={}),
            )
            group_value = generation_value_fields.setdefault(
                field.group_key,
                CfgSectionValue(fields={}),
            )
            if field.field_key in group_spec.fields:
                raise ValueError(
                    f"duplicate generation field {field.group_key}.{field.field_key}"
                )
            group_spec.fields[field.field_key] = field.spec
            group_value.fields[field.field_key] = _default_value(
                field.spec, field.default
            )
        spec_fields["generation"] = CfgSectionSpec(
            label="Generation overrides",
            fields=generation_spec_fields,
        )
        value_fields["generation"] = CfgSectionValue(fields=generation_value_fields)
        projection.update(
            {
                field.logical_key: f"generation.{field.group_key}.{field.field_key}"
                for field in generation_fields
            }
        )

    node_schema = NodeCfgSchema(
        CfgSchema(
            spec=CfgSectionSpec(
                fields=spec_fields,
                label=schema.spec.label,
                inherit_hook=schema.spec.inherit_hook,
            ),
            value=CfgSectionValue(fields=value_fields),
        ),
        logical_paths=projection,
    )
    if default_overrides:
        node_schema.with_overrides(default_overrides)
    return node_schema


def ctx_md_float(ctx: Any | None, key: str) -> float | None:
    """Return a numeric MetaDict value from ``ctx`` when present."""
    if not isinstance(ctx, ExpContext):
        return None
    value = ctx.md.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def ctx_module(ctx: Any | None, *names: str) -> Any | None:
    """Return the first ModuleLibrary module found in ``ctx`` under ``names``."""
    if not isinstance(ctx, ExpContext):
        return None
    for name in names:
        try:
            module = ctx.ml.get_module(name)
        except (KeyError, ValueError):
            module = None
        if module is not None:
            return module
    return None


def nested_get(value: Any, *path: str) -> Any | None:
    """Read a nested attr/dict path from raw dicts or cfg objects."""
    cur = value
    for part in path:
        if isinstance(cur, Mapping):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def pulse_gain(module: Any) -> float | None:
    value = nested_get(module, "gain")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def pulse_length(module: Any) -> float | None:
    value = nested_get(module, "waveform", "length")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def pulse_product(module: Any) -> float | None:
    length = pulse_length(module)
    gain = pulse_gain(module)
    if length is None or gain is None:
        return None
    return length * gain


def readout_pulse_freq(module: Any) -> float | None:
    value = nested_get(module, "pulse_cfg", "freq")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def readout_pulse_gain(module: Any) -> float | None:
    value = nested_get(module, "pulse_cfg", "gain")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def module_dict(raw_cfg: dict[str, Any], key: str) -> dict[str, Any]:
    modules = raw_cfg.get("modules")
    if not isinstance(modules, dict):
        raise RuntimeError("adapter raw cfg has no modules section")
    module = modules.get(key)
    if not isinstance(module, dict):
        raise RuntimeError(f"adapter raw cfg module {key!r} is missing")
    return deepcopy(module)


def move_module(raw_cfg: dict[str, Any], old: str, new: str) -> None:
    modules = raw_cfg.get("modules")
    if not isinstance(modules, dict):
        raise RuntimeError("adapter raw cfg has no modules section")
    if old not in modules:
        raise RuntimeError(f"adapter raw cfg module {old!r} is missing")
    modules[new] = modules.pop(old)


def sweep_range(raw_cfg: dict[str, Any], key: str) -> tuple[float, float]:
    sweep = raw_cfg.get("sweep")
    if not isinstance(sweep, dict):
        raise RuntimeError("adapter raw cfg has no sweep section")
    axis = sweep.get(key)
    start = getattr(axis, "start", None)
    stop = getattr(axis, "stop", None)
    if start is None or stop is None:
        raise RuntimeError(f"adapter raw cfg sweep {key!r} is missing start/stop")
    return (float(start), float(stop))


def _ensure_context(ctx: Any | None) -> ExpContext:
    if isinstance(ctx, ExpContext):
        return ctx
    return ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)


def _default_value(spec: ScalarSpec | SweepSpec, default: Any) -> Any:
    if isinstance(spec, SweepSpec):
        if not isinstance(default, SweepValue):
            raise TypeError(
                f"SweepSpec generation default must be SweepValue, "
                f"got {type(default).__name__}"
            )
        return default
    if isinstance(default, DirectValue):
        return default
    return DirectValue(default)
