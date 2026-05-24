from __future__ import annotations

from typing import Any

from .types import (
    CfgNodeValue,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    MultiSweepSpec,
    MultiSweepValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
)


def make_default_value(spec: CfgSectionSpec) -> CfgSectionValue:
    """Produce a default CfgSectionValue mirroring the given spec structure."""
    fields: dict[str, CfgNodeValue] = {}
    for key, node_spec in spec.fields.items():
        if isinstance(node_spec, LiteralSpec):
            fields[key] = DirectValue(node_spec.value)
        elif isinstance(node_spec, ScalarSpec):
            defaults: dict[type, Any] = {int: 0, float: 0.0, bool: False, str: ""}
            fields[key] = DirectValue(defaults.get(node_spec.type, None))
        elif isinstance(node_spec, SweepSpec):
            fields[key] = SweepValue(start=0.0, stop=1.0, expts=11)
        elif isinstance(node_spec, MultiSweepSpec):
            fields[key] = MultiSweepValue(
                axes={axis: SweepValue(0.0, 1.0, 11) for axis in node_spec.axes}
            )
        elif isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec)):
            first = node_spec.allowed[0] if node_spec.allowed else CfgSectionSpec()
            label = first.label or "Custom"
            fields[key] = (
                ModuleRefValue(f"<Custom:{label}>", make_default_value(first))
                if isinstance(node_spec, ModuleRefSpec)
                else WaveformRefValue(f"<Custom:{label}>", make_default_value(first))
            )
        elif isinstance(node_spec, CfgSectionSpec):
            fields[key] = make_default_value(node_spec)
    return CfgSectionValue(fields=fields)


def inherit_from(
    old_val: CfgSectionValue,
    old_spec: CfgSectionSpec,
    new_spec: CfgSectionSpec,
) -> CfgSectionValue:
    """Build a new CfgSectionValue from new_spec, inheriting old_val where compatible."""
    if new_spec.inherit_hook is not None:
        result = new_spec.inherit_hook(old_val, old_spec)
        if result is not None:
            return result

    new_fields: dict[str, CfgNodeValue] = {}

    for key, new_node_spec in new_spec.fields.items():
        old_node_spec = old_spec.fields.get(key)
        old_node_val = old_val.fields.get(key)

        if isinstance(new_node_spec, LiteralSpec):
            new_fields[key] = DirectValue(new_node_spec.value)
            continue

        if isinstance(new_node_spec, ScalarSpec):
            if (
                isinstance(old_node_spec, ScalarSpec)
                and old_node_spec.type is new_node_spec.type
                and isinstance(old_node_val, (DirectValue, EvalValue))
            ):
                new_fields[key] = old_node_val
            else:
                defaults: dict[type, Any] = {int: 0, float: 0.0, bool: False, str: ""}
                new_fields[key] = DirectValue(defaults.get(new_node_spec.type, None))
            continue

        if isinstance(new_node_spec, SweepSpec):
            if isinstance(old_node_spec, SweepSpec) and isinstance(
                old_node_val, SweepValue
            ):
                new_fields[key] = SweepValue(
                    old_node_val.start,
                    old_node_val.stop,
                    old_node_val.expts,
                    old_node_val.step,
                )
            else:
                new_fields[key] = SweepValue(start=0.0, stop=1.0, expts=11)
            continue

        if isinstance(new_node_spec, MultiSweepSpec):
            old_axes = (
                old_node_val.axes
                if isinstance(old_node_spec, MultiSweepSpec)
                and isinstance(old_node_val, MultiSweepValue)
                else {}
            )
            new_axes: dict[str, SweepValue] = {}
            for axis_key in new_node_spec.axes:
                if axis_key in old_axes:
                    old_sv = old_axes[axis_key]
                    new_axes[axis_key] = SweepValue(
                        old_sv.start, old_sv.stop, old_sv.expts, old_sv.step
                    )
                else:
                    new_axes[axis_key] = SweepValue(0.0, 1.0, 11)
            new_fields[key] = MultiSweepValue(axes=new_axes)
            continue

        if isinstance(new_node_spec, ModuleRefSpec):
            if isinstance(old_node_spec, ModuleRefSpec) and isinstance(
                old_node_val, ModuleRefValue
            ):
                new_fields[key] = ModuleRefValue(
                    old_node_val.chosen_key, old_node_val.value
                )
            else:
                first = (
                    new_node_spec.allowed[0]
                    if new_node_spec.allowed
                    else CfgSectionSpec()
                )
                label = first.label or "Custom"
                new_fields[key] = ModuleRefValue(
                    f"<Custom:{label}>", make_default_value(first)
                )
            continue

        if isinstance(new_node_spec, WaveformRefSpec):
            if isinstance(old_node_spec, WaveformRefSpec) and isinstance(
                old_node_val, WaveformRefValue
            ):
                new_fields[key] = WaveformRefValue(
                    old_node_val.chosen_key, old_node_val.value
                )
            else:
                first = (
                    new_node_spec.allowed[0]
                    if new_node_spec.allowed
                    else CfgSectionSpec()
                )
                label = first.label or "Custom"
                new_fields[key] = WaveformRefValue(
                    f"<Custom:{label}>", make_default_value(first)
                )
            continue

        if isinstance(new_node_spec, CfgSectionSpec):
            if isinstance(old_node_spec, CfgSectionSpec) and isinstance(
                old_node_val, CfgSectionValue
            ):
                new_fields[key] = inherit_from(
                    old_node_val, old_node_spec, new_node_spec
                )
            else:
                new_fields[key] = make_default_value(new_node_spec)
            continue

    return CfgSectionValue(fields=new_fields)
