from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from .model import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
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
    default_value_for_type,
)
from .reference_key import make_custom_reference_key, parse_custom_reference_key


def _default_reference(spec: ReferenceSpec) -> ReferenceValue | None:
    if spec.optional:
        return None
    first = spec.allowed[0]
    return ReferenceValue(
        make_custom_reference_key(first.label or "Custom"), make_default_value(first)
    )


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
            fields[key] = _default_reference(node_spec)
        elif isinstance(node_spec, CfgSectionSpec):  # pyright: ignore[reportUnnecessaryIsInstance]
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
    selected by the discriminator declared by the domain in ReferenceSpec,
    so a library value can be projected onto an adapter-local
    spec that carries extra ``LiteralSpec`` locks.
    """
    chosen = ref_val.chosen_key
    try:
        label = parse_custom_reference_key(chosen)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if label is not None:
        for spec in ref_spec.allowed:
            if (spec.label or "Custom") == label:
                return spec
        allowed = ", ".join(spec.label for spec in ref_spec.allowed)
        raise RuntimeError(
            f"Unknown custom reference label {label!r}; allowed labels: {allowed}"
        )

    disc_key = ref_spec.discriminator
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


def _input_scalar(value: DirectValue | EvalValue) -> DirectValue | EvalValue:
    if isinstance(value, EvalValue):
        return EvalValue(value.expr)
    return replace(value, validation_error=None)


def detach_input_tree(value: CfgSectionValue) -> CfgSectionValue:
    """Keep user input and linkage, not aliases or the old resolution cache."""
    candidate = deepcopy(value)

    def clear_cache(node: CfgNodeValue | None) -> CfgNodeValue | None:
        if isinstance(node, CfgSectionValue):
            node.fields = {
                key: clear_cache(child) for key, child in node.fields.items()
            }
        elif isinstance(node, ReferenceValue):
            node.resolved_label = None
            node.error = None
            clear_cache(node.value)
        elif isinstance(node, (DirectValue, EvalValue)):
            return _input_scalar(node)
        elif isinstance(node, (SweepValue, CenteredSweepValue)):
            # These are the only scalar carriers inside either range type.
            for name in ("start", "stop", "center", "span", "expts", "step"):
                part = getattr(node, name, None)
                if isinstance(part, (DirectValue, EvalValue)):
                    setattr(node, name, _input_scalar(part))
        return node

    clear_cache(candidate)
    return candidate


def _inherit_reference(
    old_spec: CfgNodeSpec | None,
    old_value: CfgNodeValue | None,
    new_spec: ReferenceSpec,
    *,
    old_disabled: bool,
) -> ReferenceValue | None:
    if not isinstance(old_spec, ReferenceSpec) or old_spec.kind != new_spec.kind:
        return _default_reference(new_spec)
    if old_value is None and old_disabled and new_spec.optional:
        return None  # preserve an explicitly disabled optional reference
    if not isinstance(old_value, ReferenceValue):
        return _default_reference(new_spec)
    old_shape = select_ref_value_spec(old_spec, old_value)
    try:
        new_shape = select_ref_value_spec(new_spec, old_value)
    except RuntimeError:
        # The new allowed set cannot represent this nested choice.
        return _default_reference(new_spec)
    return ReferenceValue(
        old_value.chosen_key,
        inherit_from(old_value.value, old_shape, new_shape),
        is_overridden=old_value.is_overridden,
    )


def _inherit_scalar(
    old_spec: CfgNodeSpec | None,
    old_value: CfgNodeValue | None,
    new_spec: ScalarSpec,
) -> DirectValue | EvalValue:
    if (
        isinstance(old_spec, ScalarSpec)
        and old_spec.type is new_spec.type
        and isinstance(old_value, (DirectValue, EvalValue))
    ):
        return old_value
    if new_spec.required or new_spec.optional:
        return DirectValue(None)
    if new_spec.choices:
        return DirectValue(new_spec.choices[0])
    return DirectValue(default_value_for_type(new_spec.type))


def _inherit_range(
    old_spec: CfgNodeSpec | None,
    old_value: CfgNodeValue | None,
    new_spec: SweepSpec | CenteredSweepSpec,
) -> SweepValue | CenteredSweepValue:
    if isinstance(new_spec, SweepSpec):
        if isinstance(old_spec, SweepSpec) and isinstance(old_value, SweepValue):
            return SweepValue(
                old_value.start, old_value.stop, old_value.expts, old_value.step
            )
        return SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)
    if isinstance(old_spec, CenteredSweepSpec) and isinstance(
        old_value, CenteredSweepValue
    ):
        return CenteredSweepValue(
            old_value.center, old_value.span, old_value.expts, old_value.step
        )
    return CenteredSweepValue(center=0.5, span=1.0, expts=11, step=0.1)


def inherit_from(
    old_val: CfgSectionValue,
    old_spec: CfgSectionSpec,
    new_spec: CfgSectionSpec,
) -> CfgSectionValue:
    """Build detached input for new_spec, inheriting compatible old input."""
    if new_spec.inherit_hook is not None:
        result = new_spec.inherit_hook(deepcopy(old_val), old_spec)
        if result is not None:
            return detach_input_tree(result)

    new_fields: dict[str, CfgNodeValue | None] = {}

    for key, new_node_spec in new_spec.fields.items():
        old_node_spec = old_spec.fields.get(key)
        old_node_val = old_val.fields.get(key)

        if isinstance(new_node_spec, LiteralSpec):
            new_fields[key] = DirectValue(new_node_spec.value)
            continue

        if isinstance(new_node_spec, ScalarSpec):
            new_fields[key] = _inherit_scalar(
                old_node_spec, old_node_val, new_node_spec
            )
            continue

        if isinstance(new_node_spec, (SweepSpec, CenteredSweepSpec)):
            new_fields[key] = _inherit_range(old_node_spec, old_node_val, new_node_spec)
            continue

        if isinstance(new_node_spec, ReferenceSpec):
            new_fields[key] = _inherit_reference(
                old_node_spec,
                old_node_val,
                new_node_spec,
                old_disabled=key in old_val.fields,
            )
            continue

        if isinstance(new_node_spec, CfgSectionSpec):  # pyright: ignore[reportUnnecessaryIsInstance]
            if isinstance(old_node_spec, CfgSectionSpec) and isinstance(
                old_node_val, CfgSectionValue
            ):
                new_fields[key] = inherit_from(
                    old_node_val, old_node_spec, new_node_spec
                )
            else:
                new_fields[key] = make_default_value(new_node_spec)
            continue

    return detach_input_tree(CfgSectionValue(fields=new_fields))
