"""Adapter-shaped default cfg schemas for autofluxdep nodes.

Fresh autofluxdep placements reuse the corresponding measure-gui adapter's spec
and copy its default value tree, then apply node-local runtime defaults before the
UI sees it. Autofluxdep adds its run-time ``generation`` section and a logical
projection for the node builder.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from zcu_tools.gui.app.autofluxdep.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ModuleRefSpec,
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

PULSE_MODULE_LEAF_PATHS: tuple[str, ...] = (
    "type",
    "ch",
    "nqz",
    "freq",
    "gain",
    "waveform",
)

READOUT_PULSE_MODULE_LEAF_PATHS: tuple[str, ...] = (
    "type",
    "pulse_cfg.ch",
    "pulse_cfg.nqz",
    "pulse_cfg.freq",
    "pulse_cfg.gain",
    "pulse_cfg.waveform",
    "ro_cfg.ro_ch",
    "ro_cfg.ro_freq",
    "ro_cfg.ro_length",
    "ro_cfg.trig_offset",
)
PULSE_READOUT_REF_LABELS: tuple[str, ...] = ("Pulse Readout",)


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
    path_renames: Mapping[str, str] | None = None,
    duplicate_paths: Mapping[str, str] | None = None,
    drop_paths: tuple[str, ...] = (),
    module_ref_labels: Mapping[str, tuple[str, ...]] | None = None,
) -> NodeCfgSchema:
    """Build a ``NodeCfgSchema`` from a copied measure-gui adapter cfg shape."""
    schema = adapter_cls().make_default_cfg(_ensure_context(ctx))
    spec_fields = dict(schema.spec.fields)
    value_fields = deepcopy(schema.value.fields)
    root_spec = CfgSectionSpec(
        fields=spec_fields,
        label=schema.spec.label,
        inherit_hook=schema.spec.inherit_hook,
    )
    root_value = CfgSectionValue(fields=value_fields)

    for source, target in (duplicate_paths or {}).items():
        _duplicate_cfg_path(root_spec, root_value, source, target)
    for source, target in (path_renames or {}).items():
        _rename_cfg_path(root_spec, root_value, source, target)
    for path in drop_paths:
        _drop_cfg_path(root_spec, root_value, path)
    for path, labels in (module_ref_labels or {}).items():
        _restrict_module_ref_labels(root_spec, path, labels)
    _prune_empty_sections(root_spec, root_value)

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
        root_spec.fields["generation"] = CfgSectionSpec(
            label="Generation overrides",
            fields=generation_spec_fields,
        )
        root_value.fields["generation"] = CfgSectionValue(
            fields=generation_value_fields
        )
        projection.update(
            {
                field.logical_key: f"generation.{field.group_key}.{field.field_key}"
                for field in generation_fields
            }
        )

    node_schema = NodeCfgSchema(
        CfgSchema(
            spec=root_spec,
            value=root_value,
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


def pop_sweep_range(
    raw_cfg: dict[str, Any], key: str, *, node_name: str
) -> tuple[float, float]:
    return pop_sweep_ranges(raw_cfg, (key,), node_name=node_name)[key]


def pop_sweep_ranges(
    raw_cfg: dict[str, Any], keys: tuple[str, ...], *, node_name: str
) -> dict[str, tuple[float, float]]:
    sweep = raw_cfg.pop("sweep", None)
    if not isinstance(sweep, dict):
        raise RuntimeError(f"{node_name} raw cfg has no sweep section")
    ranges: dict[str, tuple[float, float]] = {}
    for key in keys:
        if key not in sweep:
            raise RuntimeError(f"{node_name} raw cfg has no sweep.{key}")
        ranges[key] = _raw_range_tuple(sweep[key])
    return ranges


def _raw_range_tuple(value: Any) -> tuple[float, float]:
    if hasattr(value, "start") and hasattr(value, "stop"):
        return (float(value.start), float(value.stop))
    lo, hi = value
    return (float(lo), float(hi))


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


def _duplicate_cfg_path(
    root_spec: CfgSectionSpec,
    root_value: CfgSectionValue,
    source: str,
    target: str,
) -> None:
    _set_cfg_node(
        root_spec,
        target,
        deepcopy(_get_cfg_node(root_spec, source)),
        subject="spec",
    )
    _set_cfg_node(
        root_value,
        target,
        deepcopy(_get_cfg_node(root_value, source)),
        subject="value",
    )


def _rename_cfg_path(
    root_spec: CfgSectionSpec,
    root_value: CfgSectionValue,
    source: str,
    target: str,
) -> None:
    _set_cfg_node(
        root_spec,
        target,
        _pop_cfg_node(root_spec, source),
        subject="spec",
    )
    _set_cfg_node(
        root_value,
        target,
        _pop_cfg_node(root_value, source),
        subject="value",
    )


def _drop_cfg_path(
    root_spec: CfgSectionSpec,
    root_value: CfgSectionValue,
    path: str,
) -> None:
    _pop_cfg_node(root_spec, path)
    _pop_cfg_node(root_value, path)


def _restrict_module_ref_labels(
    root_spec: CfgSectionSpec,
    path: str,
    labels: tuple[str, ...],
) -> None:
    node = _get_cfg_node(root_spec, path)
    if not isinstance(node, ModuleRefSpec):
        raise TypeError(f"cfg spec path {path!r} is not a ModuleRefSpec")
    allowed_labels = set(labels)
    allowed = [spec for spec in node.allowed if spec.label in allowed_labels]
    if not allowed:
        available = ", ".join(spec.label for spec in node.allowed)
        wanted = ", ".join(labels)
        raise ValueError(
            f"cfg spec path {path!r} has no allowed ModuleRef labels matching "
            f"{wanted!r}; available: {available}"
        )
    _replace_cfg_node(root_spec, path, replace(node, allowed=allowed))


def _get_cfg_node(root: CfgSectionSpec | CfgSectionValue, path: str) -> Any:
    node: Any = root
    for part in _split_path(path):
        fields = getattr(node, "fields", None)
        if not isinstance(fields, dict) or part not in fields:
            raise KeyError(f"cfg path {path!r} segment {part!r} is missing")
        node = fields[part]
    return node


def _pop_cfg_node(root: CfgSectionSpec | CfgSectionValue, path: str) -> Any:
    parts = _split_path(path)
    parent: Any = root
    for part in parts[:-1]:
        fields = getattr(parent, "fields", None)
        if not isinstance(fields, dict) or part not in fields:
            raise KeyError(f"cfg path {path!r} segment {part!r} is missing")
        parent = fields[part]
    fields = getattr(parent, "fields", None)
    if not isinstance(fields, dict) or parts[-1] not in fields:
        raise KeyError(f"cfg path {path!r} leaf {parts[-1]!r} is missing")
    return fields.pop(parts[-1])


def _replace_cfg_node(
    root: CfgSectionSpec | CfgSectionValue,
    path: str,
    node: Any,
) -> None:
    parts = _split_path(path)
    parent: Any = root
    for part in parts[:-1]:
        fields = getattr(parent, "fields", None)
        if not isinstance(fields, dict) or part not in fields:
            raise KeyError(f"cfg path {path!r} segment {part!r} is missing")
        parent = fields[part]
    fields = getattr(parent, "fields", None)
    if not isinstance(fields, dict) or parts[-1] not in fields:
        raise KeyError(f"cfg path {path!r} leaf {parts[-1]!r} is missing")
    fields[parts[-1]] = node


def _set_cfg_node(
    root: CfgSectionSpec | CfgSectionValue,
    path: str,
    node: Any,
    *,
    subject: str,
) -> None:
    parts = _split_path(path)
    parent: Any = root
    section_cls = (
        CfgSectionSpec if isinstance(root, CfgSectionSpec) else CfgSectionValue
    )
    for part in parts[:-1]:
        fields = getattr(parent, "fields", None)
        if not isinstance(fields, dict):
            raise TypeError(f"cfg {subject} path {path!r} cannot descend into {part!r}")
        child = fields.get(part)
        if child is None:
            child = (
                CfgSectionSpec(fields={}, label=part)
                if section_cls is CfgSectionSpec
                else CfgSectionValue(fields={})
            )
            fields[part] = child
        parent = child
    fields = getattr(parent, "fields", None)
    if not isinstance(fields, dict):
        raise TypeError(f"cfg {subject} path {path!r} cannot assign leaf")
    if parts[-1] in fields:
        raise ValueError(f"cfg {subject} path {path!r} already exists")
    fields[parts[-1]] = node


def _prune_empty_sections(spec: CfgSectionSpec, value: CfgSectionValue) -> bool:
    for key in list(spec.fields):
        child_spec = spec.fields[key]
        child_value = value.fields.get(key)
        if isinstance(child_spec, CfgSectionSpec) and isinstance(
            child_value, CfgSectionValue
        ):
            if _prune_empty_sections(child_spec, child_value):
                del spec.fields[key]
                value.fields.pop(key, None)
    return not spec.fields


def _split_path(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split(".") if part)
    if not parts:
        raise ValueError("cfg path must not be empty")
    return parts
