from __future__ import annotations

import logging

from .model import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeValue,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    _reference_discriminator_key,
    default_value_for_type,
)
from .reference_key import make_custom_reference_key, parse_custom_reference_key

logger = logging.getLogger(__name__)


def make_default_value(spec: CfgSectionSpec) -> CfgSectionValue:
    """Produce a default CfgSectionValue mirroring the given spec structure.

    A structural helper for cfg construction: it guesses sensible defaults
    (scalar 0, sweep range, choices[0]) so callers need not spell out every field.
    Domain-specific defaults remain outside this helper. The
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
        elif isinstance(node_spec, CenteredSweepSpec):
            fields[key] = CenteredSweepValue(center=0.5, span=1.0, expts=11, step=0.1)
        elif isinstance(node_spec, ReferenceSpec):
            if node_spec.optional:
                fields[key] = None  # optional ref defaults to disabled (ADR-0010)
            else:
                first = node_spec.allowed[0]
                label = first.label or "Custom"
                fields[key] = ReferenceValue(
                    make_custom_reference_key(label), make_default_value(first)
                )
        elif isinstance(node_spec, CfgSectionSpec):
            fields[key] = make_default_value(node_spec)
        else:
            raise TypeError(
                f"Unsupported cfg spec node {type(node_spec).__name__} at {key!r}"
            )
    return CfgSectionValue(fields=fields)


def select_ref_value_spec(
    ref_spec: ReferenceSpec,
    ref_val: ReferenceValue,
) -> CfgSectionSpec:
    """Return the caller-allowed spec matching a ref value's concrete shape.

    Custom refs are selected by their ``<Custom:label>`` key. Linked refs are
    selected by the value tree's discriminator (``type`` for modules, ``style``
    for waveforms), so a library value can be projected onto an adapter-local
    spec that carries extra ``LiteralSpec`` locks.
    """
    chosen = ref_val.chosen_key
    try:
        label = parse_custom_reference_key(chosen)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if label is not None:
        for spec in ref_spec.allowed:
            if spec.label == label:
                return spec
        allowed = ", ".join(spec.label for spec in ref_spec.allowed)
        raise RuntimeError(
            f"Unknown custom reference label {label!r}; allowed labels: {allowed}"
        )

    disc_key = _reference_discriminator_key(ref_spec)
    disc_label = disc_key or "discriminator"
    discriminator = _section_discriminator(ref_val.value, disc_key)
    if discriminator is None:
        if len(ref_spec.allowed) == 1:
            return ref_spec.allowed[0]
        available = ", ".join(
            repr(getattr(spec.fields.get(disc_label), "value", None))
            for spec in ref_spec.allowed
        )
        raise RuntimeError(
            f"Reference {chosen!r} has no {disc_label!r} discriminator; "
            f"available {disc_label}: {available}"
        )

    for spec in ref_spec.allowed:
        leaf = spec.fields.get(disc_label)
        if isinstance(leaf, LiteralSpec) and leaf.value == discriminator:
            return spec

    available = ", ".join(
        repr(getattr(spec.fields.get(disc_label), "value", None))
        for spec in ref_spec.allowed
    )
    raise RuntimeError(
        f"Reference {chosen!r} has {disc_label}={discriminator!r}, but no allowed "
        f"shape matches (available {disc_label}: {available})"
    )


def align_locked_literals(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
) -> CfgSectionValue:
    """Mutate ``value`` so every ``LiteralSpec`` leaf matches ``spec.value``.

    This is a projection step, not validation: callers use it when a value tree
    built outside the adapter-local spec (for example a linked ModuleLibrary
    entry) enters that spec. Non-literal inconsistencies remain for validate()
    to catch.
    """
    for key, node_spec in spec.fields.items():
        node_val = value.fields.get(key)
        if isinstance(node_spec, LiteralSpec):
            value.fields[key] = DirectValue(node_spec.value)
        elif isinstance(node_spec, CfgSectionSpec) and isinstance(
            node_val, CfgSectionValue
        ):
            align_locked_literals(node_spec, node_val)
        elif isinstance(node_spec, ReferenceSpec) and isinstance(
            node_val, ReferenceValue
        ):
            chosen_spec = select_ref_value_spec(node_spec, node_val)
            align_locked_literals(chosen_spec, node_val.value)
    return value


def _section_discriminator(value: CfgSectionValue, key: str | None) -> object:
    leaf = value.fields.get(key) if key is not None else None
    return getattr(leaf, "value", None)


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

        if isinstance(new_node_spec, CenteredSweepSpec):
            if isinstance(old_node_spec, CenteredSweepSpec) and isinstance(
                old_node_val, CenteredSweepValue
            ):
                new_fields[key] = CenteredSweepValue(
                    old_node_val.center,
                    old_node_val.span,
                    old_node_val.expts,
                    old_node_val.step,
                )
            else:
                new_fields[key] = CenteredSweepValue(
                    center=0.5, span=1.0, expts=11, step=0.1
                )
            continue

        if isinstance(new_node_spec, ReferenceSpec):
            if (
                isinstance(old_node_spec, ReferenceSpec)
                and old_node_spec.kind == new_node_spec.kind
                and isinstance(old_node_val, ReferenceValue)
            ):
                new_fields[key] = ReferenceValue(
                    old_node_val.chosen_key, old_node_val.value
                )
            elif (
                isinstance(old_node_spec, ReferenceSpec)
                and old_node_spec.kind == new_node_spec.kind
                and key in old_val.fields
                and old_node_val is None
            ):
                new_fields[key] = None  # inherit the disabled state (ADR-0010)
            elif new_node_spec.optional:
                new_fields[key] = None  # optional ref defaults to disabled
            else:
                first = new_node_spec.allowed[0]
                label = first.label or "Custom"
                new_fields[key] = ReferenceValue(
                    make_custom_reference_key(label), make_default_value(first)
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
