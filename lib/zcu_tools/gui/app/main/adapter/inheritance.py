from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from .types import (
    CfgNodeValue,
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
    default_value_for_type,
)


def make_default_value(spec: CfgSectionSpec) -> CfgSectionValue:
    """Produce a default CfgSectionValue mirroring the given spec structure.

    A helper for adapters' ``make_default_value(ctx)``: it guesses sensible
    defaults (scalar 0, sweep range, choices[0]) so an adapter need not spell out
    every field — special cases are overridden via the value OO fluent. The
    result is **complete**: every spec field has an entry, no missing keys
    (ADR-0010). An *optional* ModuleRef/WaveformRef defaults to ``None``
    (disabled) — the safest, least-surprising default for "this field is
    optional"; an adapter that wants it enabled supplies a ref factory value.
    """
    fields: dict[str, CfgNodeValue | None] = {}
    for key, node_spec in spec.fields.items():
        if isinstance(node_spec, LiteralSpec):
            fields[key] = DirectValue(node_spec.value)
        elif isinstance(node_spec, ScalarSpec):
            if node_spec.required or node_spec.optional:
                fields[key] = DirectValue(value=None)  # unset (ADR-0010)
            elif node_spec.choices:
                fields[key] = DirectValue(node_spec.choices[0])
            else:
                fields[key] = DirectValue(default_value_for_type(node_spec.type))
        elif isinstance(node_spec, SweepSpec):
            fields[key] = SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)
        elif isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec)):
            if node_spec.optional:
                fields[key] = None  # optional ref defaults to disabled (ADR-0010)
            else:
                first = node_spec.allowed[0]
                label = first.label or "Custom"
                fields[key] = (
                    ModuleRefValue(f"<Custom:{label}>", make_default_value(first))
                    if isinstance(node_spec, ModuleRefSpec)
                    else WaveformRefValue(
                        f"<Custom:{label}>", make_default_value(first)
                    )
                )
        elif isinstance(node_spec, DeviceRefSpec):
            fields[key] = DirectValue("")
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

    new_fields: dict[str, CfgNodeValue | None] = {}

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
            elif new_node_spec.required or new_node_spec.optional:
                new_fields[key] = DirectValue(value=None)  # unset (ADR-0010)
            elif new_node_spec.choices:
                new_fields[key] = DirectValue(new_node_spec.choices[0])
            else:
                new_fields[key] = DirectValue(
                    default_value_for_type(new_node_spec.type)
                )
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
                new_fields[key] = SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)
            continue

        if isinstance(new_node_spec, ModuleRefSpec):
            if isinstance(old_node_spec, ModuleRefSpec) and isinstance(
                old_node_val, ModuleRefValue
            ):
                new_fields[key] = ModuleRefValue(
                    old_node_val.chosen_key, old_node_val.value
                )
            elif (
                isinstance(old_node_spec, ModuleRefSpec)
                and key in old_val.fields
                and old_node_val is None
            ):
                new_fields[key] = None  # inherit the disabled state (ADR-0010)
            elif new_node_spec.optional:
                new_fields[key] = None  # optional ref defaults to disabled
            else:
                first = new_node_spec.allowed[0]
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
            elif (
                isinstance(old_node_spec, WaveformRefSpec)
                and key in old_val.fields
                and old_node_val is None
            ):
                new_fields[key] = None  # inherit the disabled state (ADR-0010)
            elif new_node_spec.optional:
                new_fields[key] = None  # optional ref defaults to disabled
            else:
                first = new_node_spec.allowed[0]
                label = first.label or "Custom"
                new_fields[key] = WaveformRefValue(
                    f"<Custom:{label}>", make_default_value(first)
                )
            continue

        if isinstance(new_node_spec, DeviceRefSpec):
            if isinstance(old_node_spec, DeviceRefSpec) and isinstance(
                old_node_val, DirectValue
            ):
                new_fields[key] = old_node_val
            else:
                new_fields[key] = DirectValue("")
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
