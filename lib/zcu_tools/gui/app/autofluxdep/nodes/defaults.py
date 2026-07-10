"""Legacy schema compatibility and Default cfg patch helpers for nodes.

Experiment nodes build fresh schemas with ``nodes.utils.NodeSchemaBuilder``.
This module keeps older adapter-shaped helpers for compatibility tests and owns
the shared patch helpers that translate concrete dependency modules into
declared Default cfg leaf patches.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from zcu_tools.gui.app.autofluxdep.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ChoiceBinding,
    ChoiceSectionSpec,
    DirectValue,
    ModuleRefSpec,
    ModuleRefValue,
    NodeCfgSchema,
    OverridePath,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    align_locked_literals,
    module_cfg_to_value,
    module_leaf_patches,
    module_override_paths,
)
from zcu_tools.gui.session.types import ExpContext
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from .utils import (
    ctx_md_float,
    ctx_module,
    nested_get,
    pulse_gain,
    pulse_length,
    pulse_product,
    readout_pulse_freq,
    readout_pulse_gain,
)
from .utils.override_plan import PULSE_MODULE_LEAF_PATHS, READOUT_FALLBACK_LEAF_PATHS

GENERATION_GROUP_LABELS: dict[str, str] = {
    "acquisition": "Acquisition guardrails",
    "drive_gain": "Drive-gain adaptation",
    "freq_recovery": "Frequency recovery",
    "pi_feedback": "Pi-length feedback",
    "predictor_correction": "Predictor correction",
    "relax": "Relax timing",
    "search_center": "Readout search center",
    "search_window": "Readout search window",
    # Older/generic groups remain available for helper tests and future nodes; real
    # builders choose domain-specific groups when a clearer label exists.
    "sweep": "Sweep generation",
    "timing": "Timing / relax",
    "feedback": "Feedback / adaptive",
    "fit": "Fit behavior",
    "freq_search": "Readout frequency search",
    "gain_search": "Readout gain search",
    "safety": "Safety gates",
}

PULSE_READOUT_REF_LABELS: tuple[str, ...] = ("Pulse Readout",)


@dataclass(frozen=True)
class GenerationField:
    logical_key: str
    field_key: str
    spec: ScalarSpec | SweepSpec | CenteredSweepSpec
    default: Any
    group_key: str
    group_label: str


@dataclass(frozen=True)
class GenerationChoice:
    group_key: str
    selector_key: str
    choices: Mapping[str, tuple[str, ...]]


def generation_field(
    logical_key: str,
    field_key: str,
    spec: ScalarSpec | SweepSpec | CenteredSweepSpec,
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


def logical_generation_field(
    key: str,
    spec: ScalarSpec | SweepSpec | CenteredSweepSpec,
    default: Any,
    *,
    group: str,
    group_label: str | None = None,
) -> GenerationField:
    return generation_field(
        key,
        key,
        spec,
        default,
        group=group,
        group_label=group_label,
    )


def generation_choice(
    group: str,
    selector: str,
    choices: Mapping[str, tuple[str, ...]],
) -> GenerationChoice:
    if not group:
        raise ValueError("generation choice group must be non-empty")
    if not selector:
        raise ValueError("generation choice selector must be non-empty")
    if not choices:
        raise ValueError("generation choice needs at least one choice")
    return GenerationChoice(group, selector, choices)


def pulse_module_override_paths(
    module_name: str, *, source: str, reason: str
) -> tuple[OverridePath, ...]:
    return module_override_paths(
        prefix=f"modules.{module_name}",
        leaf_paths=PULSE_MODULE_LEAF_PATHS,
        source=source,
        reason=reason,
    )


def readout_module_override_paths(
    *, source: str, reason: str
) -> tuple[OverridePath, ...]:
    return module_override_paths(
        prefix="modules.readout",
        leaf_paths=READOUT_FALLBACK_LEAF_PATHS,
        source=source,
        reason=reason,
        mode="fallback",
    )


def pulse_module_patches(module_name: str, module: object) -> dict[str, object]:
    return module_leaf_patches(
        prefix=f"modules.{module_name}",
        module=module,
        leaf_paths=PULSE_MODULE_LEAF_PATHS,
    )


def readout_module_patches(readout: object) -> dict[str, object]:
    return module_leaf_patches(
        prefix="modules.readout",
        module=readout,
        leaf_paths=READOUT_FALLBACK_LEAF_PATHS,
    )


def adapter_node_schema(
    adapter_cls: type[Any],
    ctx: Any | None,
    *,
    logical_paths: Mapping[str, str],
    generation_fields: tuple[GenerationField, ...] = (),
    spec_overrides: Mapping[str, CfgNodeSpec] | None = None,
    default_overrides: Mapping[str, Any] | None = None,
    path_renames: Mapping[str, str] | None = None,
    duplicate_paths: Mapping[str, str] | None = None,
    drop_paths: tuple[str, ...] = (),
    module_ref_labels: Mapping[str, tuple[str, ...]] | None = None,
    label_overrides: Mapping[str, str] | None = None,
    generation_choices: tuple[GenerationChoice, ...] = (),
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
    for path, spec_node in (spec_overrides or {}).items():
        _replace_cfg_node(root_spec, path, spec_node)
    for path, label in (label_overrides or {}).items():
        _relabel_cfg_node(root_spec, path, label)
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
        if generation_choices:
            choices_by_group: dict[str, list[ChoiceBinding]] = {}
            for choice in generation_choices:
                group_spec = generation_spec_fields.get(choice.group_key)
                if not isinstance(group_spec, CfgSectionSpec):
                    raise ValueError(
                        f"generation choice group {choice.group_key!r} is not declared"
                    )
                choices_by_group.setdefault(choice.group_key, []).append(
                    _generation_choice_binding(group_spec, choice)
                )
            for group_key, bindings in choices_by_group.items():
                group_spec = generation_spec_fields[group_key]
                assert isinstance(group_spec, CfgSectionSpec)
                generation_spec_fields[group_key] = ChoiceSectionSpec(
                    fields=dict(group_spec.fields),
                    label=group_spec.label,
                    inherit_hook=group_spec.inherit_hook,
                    bindings=tuple(bindings),
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
        align_locked_literals(node_schema.schema.spec, node_schema.schema.value)
    return node_schema


def _generation_choice_binding(
    group_spec: CfgSectionSpec, choice: GenerationChoice
) -> ChoiceBinding:
    variants: dict[str, CfgSectionSpec] = {}
    for value, field_keys in choice.choices.items():
        fields: dict[str, Any] = {}
        for key in field_keys:
            try:
                fields[key] = group_spec.fields[key]
            except KeyError as exc:
                raise ValueError(
                    f"generation choice {choice.group_key}.{choice.selector_key}="
                    f"{value!r} references unknown field {key!r}"
                ) from exc
        variants[value] = CfgSectionSpec(fields=fields, label=value)
    return ChoiceBinding(choice.selector_key, variants)


def module_ref_value_from_ctx(ctx: Any | None, *names: str) -> ModuleRefValue | None:
    """Return a linked module ref for the first named module present in ``ctx``."""
    if not isinstance(ctx, ExpContext):
        return None
    for name in names:
        try:
            module = ctx.ml.get_module(name)
        except (KeyError, ValueError):
            module = None
        if module is None:
            continue
        _, value = module_cfg_to_value(module)
        return ModuleRefValue(chosen_key=name, value=value)
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


def _default_value(
    spec: ScalarSpec | SweepSpec | CenteredSweepSpec, default: Any
) -> Any:
    if isinstance(spec, SweepSpec):
        if not isinstance(default, SweepValue):
            raise TypeError(
                f"SweepSpec generation default must be SweepValue, "
                f"got {type(default).__name__}"
            )
        return default
    if isinstance(spec, CenteredSweepSpec):
        if not isinstance(default, CenteredSweepValue):
            raise TypeError(
                "CenteredSweepSpec generation default must be CenteredSweepValue, "
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


def _relabel_cfg_node(root_spec: CfgSectionSpec, path: str, label: str) -> None:
    node = _get_cfg_node(root_spec, path)
    if not hasattr(node, "label"):
        raise TypeError(f"cfg spec path {path!r} does not have a label")
    _replace_cfg_node(root_spec, path, replace(node, label=label))


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
