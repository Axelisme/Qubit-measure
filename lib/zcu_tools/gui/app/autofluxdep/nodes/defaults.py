"""Adapter-backed default cfg schemas for autofluxdep nodes.

Fresh autofluxdep placements reuse the corresponding measure-gui adapter's
``make_default_cfg(ctx)`` result as the visible Default cfg. Autofluxdep only adds
its run-time ``generation`` section and a logical projection for the node builder.
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


@dataclass(frozen=True)
class GenerationField:
    logical_key: str
    field_key: str
    spec: ScalarSpec | SweepSpec
    default: Any


def generation_field(
    logical_key: str, field_key: str, spec: ScalarSpec | SweepSpec, default: Any
) -> GenerationField:
    return GenerationField(
        logical_key=logical_key, field_key=field_key, spec=spec, default=default
    )


def adapter_node_schema(
    adapter_cls: type[Any],
    ctx: Any | None,
    *,
    logical_paths: Mapping[str, str],
    generation_fields: tuple[GenerationField, ...] = (),
) -> NodeCfgSchema:
    """Build a ``NodeCfgSchema`` from a measure-gui adapter default cfg."""
    schema = adapter_cls().make_default_cfg(_ensure_context(ctx))
    spec_fields = dict(schema.spec.fields)
    value_fields = dict(schema.value.fields)
    projection = dict(logical_paths)

    if generation_fields:
        spec_fields["generation"] = CfgSectionSpec(
            label="Generation overrides",
            fields={field.field_key: field.spec for field in generation_fields},
        )
        value_fields["generation"] = CfgSectionValue(
            fields={
                field.field_key: _default_value(field.spec, field.default)
                for field in generation_fields
            }
        )
        projection.update(
            {
                field.logical_key: f"generation.{field.field_key}"
                for field in generation_fields
            }
        )

    return NodeCfgSchema(
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
