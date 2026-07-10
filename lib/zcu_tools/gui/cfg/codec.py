"""Pure raw↔live transforms for shared GUI cfg persistence.

``schema_to_raw`` encodes a live ``CfgSchema`` as a JSON-compatible dict;
``raw_to_schema`` rebuilds the value tree against a base spec. The codec owns no
I/O or app state, and both measure workspace persistence and autofluxdep node
persistence use the same wire shape.
"""

from __future__ import annotations

from collections.abc import Iterable

from .inheritance import make_default_value
from .model import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    _reference_discriminator_key,
)


class SessionCodecError(RuntimeError):
    """Invalid persisted cfg payload encountered while rebuilding a live value."""


def _to_json_compatible(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_json_compatible(to_dict())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_json_compatible(model_dump())
    return str(value)


def schema_to_raw(schema: CfgSchema) -> dict[str, object]:
    """Lower a live cfg schema to a JSON-able raw dict."""
    return _section_value_to_raw(schema.spec, schema.value)


def raw_to_schema(base_schema: CfgSchema, raw_cfg: dict[str, object]) -> CfgSchema:
    """Rebuild a live cfg schema from a raw dict against the adapter base spec."""
    try:
        value = _section_value_from_raw(base_schema.spec, raw_cfg)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise SessionCodecError(f"Invalid session cfg payload: {exc}") from exc
    return CfgSchema(spec=base_schema.spec, value=value)


def _section_value_to_raw(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    *,
    fill_missing_literals: bool = False,
) -> dict[str, object]:
    # The value tree is complete for user-editable fields. Literal fields may be
    # omitted by form refresh paths because the spec is canonical; optional refs
    # use ``None`` for the disabled marker (ADR-0010).
    payload: dict[str, object] = {}
    for key, node_spec in spec.fields.items():
        node_val = value.fields.get(key)
        if node_val is None:
            if isinstance(node_spec, LiteralSpec) and fill_missing_literals:
                payload[key] = _node_value_to_raw(node_spec, DirectValue(None))
            elif isinstance(node_spec, ReferenceSpec):
                payload[key] = {"__kind": "disabled"}
            else:
                raise SessionCodecError(
                    f"Value field {key!r} is None for a non-ref spec "
                    f"{type(node_spec).__name__} (incomplete value tree)"
                )
        else:
            payload[key] = _node_value_to_raw(
                node_spec,
                node_val,
                fill_missing_literals=fill_missing_literals,
            )
    return payload


def _node_value_to_raw(
    spec: CfgNodeSpec,
    value: CfgNodeValue,
    *,
    fill_missing_literals: bool = False,
) -> object:
    if isinstance(spec, LiteralSpec):
        # Fixed-value field; the literal value is canonical.
        return {"__kind": "direct", "value": _to_json_compatible(spec.value)}
    if isinstance(spec, ScalarSpec):
        assert isinstance(value, (DirectValue, EvalValue))
        if isinstance(value, EvalValue):
            return {
                "__kind": "eval",
                "expr": value.expr,
            }
        return {"__kind": "direct", "value": _to_json_compatible(value.value)}
    if isinstance(spec, SweepSpec):
        assert isinstance(value, SweepValue)
        return {
            "start": _sweep_edge_to_raw(value.start),
            "stop": _sweep_edge_to_raw(value.stop),
            "expts": value.expts,
            "step": value.step,
        }
    if isinstance(spec, CenteredSweepSpec):
        assert isinstance(value, CenteredSweepValue)
        return {
            "center": _sweep_edge_to_raw(value.center),
            "span": value.span,
            "expts": value.expts,
            "step": value.step,
        }
    if isinstance(spec, DeviceRefSpec):
        assert isinstance(value, DirectValue)
        return {"__kind": "direct", "value": _to_json_compatible(value.value)}
    if isinstance(spec, CfgSectionSpec):
        assert isinstance(value, CfgSectionValue)
        return _section_value_to_raw(
            spec,
            value,
            fill_missing_literals=fill_missing_literals,
        )
    if isinstance(spec, ReferenceSpec):
        assert isinstance(value, ReferenceValue)
        disc_key = _reference_discriminator_key(spec)
        return {
            "__kind": f"{spec.kind}_ref",
            "chosen_key": value.chosen_key,
            "is_overridden": value.is_overridden,
            "value": _section_value_to_raw(
                _select_allowed_spec(
                    spec,
                    value.chosen_key,
                    _value_discriminator(value.value, disc_key),
                    value_fields=value.value.fields.keys(),
                ),
                value.value,
                fill_missing_literals=True,
            ),
        }
    return _to_json_compatible(value)


def _sweep_edge_to_raw(value: float | EvalValue) -> object:
    if isinstance(value, EvalValue):
        return {"__kind": "eval", "expr": value.expr}
    return float(value)


def _section_value_from_raw(
    spec: CfgSectionSpec,
    raw: dict[str, object],
) -> CfgSectionValue:
    # Start from the complete default (ADR-0010); override each key present in
    # ``raw``. A ``{"__kind": "disabled"}`` marker → ``None`` (disabled). A key
    # genuinely absent from ``raw`` (old file / new field) keeps the default —
    # which for an optional ref is already ``None`` (disabled), faithfully
    # restoring the disabled state.
    value = make_default_value(spec)
    for key, node_spec in spec.fields.items():
        if key not in raw:
            continue
        raw_node = raw[key]
        if isinstance(raw_node, dict) and raw_node.get("__kind") == "disabled":
            value.fields[key] = None
        else:
            value.fields[key] = _node_value_from_raw(node_spec, raw_node)
    return value


def _node_value_from_raw(
    spec: CfgNodeSpec,
    raw: object,
) -> CfgNodeValue:
    if isinstance(spec, LiteralSpec):
        # Locked field: the value is canonical from the spec, ignore the payload.
        return DirectValue(spec.value)
    if isinstance(spec, ScalarSpec):
        if (
            isinstance(raw, dict)
            and raw.get("__kind") == "eval"
            and isinstance(raw.get("expr"), str)
        ):
            return EvalValue(expr=raw["expr"], resolved=None, error=None)
        if isinstance(raw, dict) and raw.get("__kind") == "direct":
            # ``value`` is None when the scalar is unset (ADR-0010).
            return DirectValue(value=raw.get("value"))
        if isinstance(raw, str) and raw.strip().startswith("="):
            raise RuntimeError("Legacy scalar '=expr' payload is unsupported")
        return DirectValue(raw)
    if isinstance(spec, SweepSpec):
        if isinstance(raw, dict):
            start = _parse_sweep_edge(raw["start"])
            stop = _parse_sweep_edge(raw["stop"])
            expts = int(raw["expts"])
            step_raw = raw.get("step")
            if step_raw is None:
                raise RuntimeError("Sweep step is required in session payload")
            step = float(step_raw)
            return SweepValue(start=start, stop=stop, expts=expts, step=step)
        raise RuntimeError("Sweep payload must be an object")
    if isinstance(spec, CenteredSweepSpec):
        if isinstance(raw, dict):
            center = _parse_sweep_edge(raw["center"])
            span = float(raw["span"])
            expts = int(raw["expts"])
            step_raw = raw.get("step")
            if step_raw is None:
                raise RuntimeError("Centered sweep step is required in session payload")
            step = float(step_raw)
            return CenteredSweepValue(
                center=center,
                span=span,
                expts=expts,
                step=step,
            )
        raise RuntimeError("Centered sweep payload must be an object")
    if isinstance(spec, DeviceRefSpec):
        if isinstance(raw, dict) and raw.get("__kind") == "direct":
            value = raw.get("value")
            if not isinstance(value, str):
                raise RuntimeError("Device reference value must be string")
            return DirectValue(value)
        raise RuntimeError("Device reference must use direct payload encoding")
    if isinstance(spec, CfgSectionSpec):
        if not isinstance(raw, dict):
            raise RuntimeError("Section payload must be an object")
        return _section_value_from_raw(spec, raw)
    if isinstance(spec, ReferenceSpec):
        return _ref_value_from_raw(spec, raw)
    raise RuntimeError(f"Unsupported spec node for restore: {type(spec).__name__}")


def _parse_sweep_edge(raw: object) -> float | EvalValue:
    if (
        isinstance(raw, dict)
        and raw.get("__kind") == "eval"
        and isinstance(raw.get("expr"), str)
    ):
        return EvalValue(expr=raw["expr"], resolved=None, error=None)
    if isinstance(raw, str) and raw.strip().startswith("="):
        raise RuntimeError("Legacy sweep '=expr' payload is unsupported")
    if isinstance(raw, (int, float)):
        return float(raw)
    raise RuntimeError("Sweep edge must be numeric or '=expr'")


def _ref_value_from_raw(
    spec: ReferenceSpec,
    raw: object,
) -> ReferenceValue:
    # The disabled marker is handled by the caller (``_section_value_from_raw``
    # maps it to None) — this only decodes an enabled ref.
    if (
        isinstance(raw, dict)
        and raw.get("__kind") == f"{spec.kind}_ref"
        and isinstance(raw.get("chosen_key"), str)
        and isinstance(raw.get("value"), dict)
    ):
        chosen_key = raw["chosen_key"]
        raw_value = raw["value"]
        disc_key = _reference_discriminator_key(spec)
        value_spec = _select_allowed_spec(
            spec,
            chosen_key,
            _raw_discriminator(raw_value, disc_key),
            value_fields=raw_value.keys(),
        )
        nested = _section_value_from_raw(value_spec, raw_value)
        return ReferenceValue(
            chosen_key=chosen_key,
            value=nested,
            is_overridden=bool(raw.get("is_overridden", False)),
        )
    kind_label = spec.kind.capitalize()
    raise RuntimeError(
        f"{kind_label} reference must use {spec.kind}_ref payload encoding"
    )


def _value_discriminator(value: CfgSectionValue, key: str | None) -> object:
    """The discriminator leaf's value from a live section value (``None`` if the
    field is absent). ``key`` is ``type`` for modules / ``style`` for waveforms;
    the leaf is a ``DirectValue`` whose ``.value`` names the chosen shape."""
    leaf = value.fields.get(key) if key is not None else None
    return getattr(leaf, "value", None)


def _raw_discriminator(raw_value: object, key: str | None) -> object:
    """The discriminator leaf's value from a raw section dict (``None`` if absent).
    Mirror of ``_value_discriminator`` for the serialised side."""
    if isinstance(raw_value, dict) and key is not None:
        node = raw_value.get(key)
        if isinstance(node, dict):
            return node.get("value")
    return None


def _select_allowed_spec(
    spec: ReferenceSpec,
    chosen_key: str,
    discriminator: object,
    *,
    value_fields: object | None = None,
) -> CfgSectionSpec:
    """Pick the allowed shape a ref's value actually uses (both directions).

    A ``<Custom:Label>`` key names the shape by label. A LINKED (library-named)
    key does not — but the value/raw normally carries the shape's discriminator
    leaf (``type`` for module refs, ``style`` for waveform refs, each locked to a
    distinct ``LiteralSpec``). Some live form refresh paths omit locked literal
    leaves, so the shape can also be recovered from the non-literal field set
    when that is unambiguous. Both routes keep this codec pure, unlike
    finished-cfg reference selection, which consults a live library resolver.

    Fast-fails on no match rather than silently defaulting to ``allowed[0]`` —
    that default mis-shaped a multi-shape ref (e.g. a readout LINKED to a library
    pulse-readout but serialised against the first allowed direct-readout shape,
    which then crashes on the missing ``ro_ch`` field)."""
    if chosen_key.startswith("<Custom:") and chosen_key.endswith(">"):
        label = chosen_key[len("<Custom:") : -1]
        for allowed_spec in spec.allowed:
            if allowed_spec.label == label:
                return allowed_spec
    # A single allowed shape is unambiguous — no discriminator needed (and the
    # shape need not even carry one).
    if len(spec.allowed) == 1:
        return spec.allowed[0]
    disc_key = _reference_discriminator_key(spec)
    disc_label = disc_key or "discriminator"
    for allowed_spec in spec.allowed:
        leaf = allowed_spec.fields.get(disc_label)
        if isinstance(leaf, LiteralSpec) and leaf.value == discriminator:
            return allowed_spec
    if discriminator is None and value_fields is not None:
        by_fields = _select_allowed_spec_by_fields(spec, disc_label, value_fields)
        if by_fields is not None:
            return by_fields
    available = ", ".join(
        repr(getattr(s.fields.get(disc_label), "value", None)) for s in spec.allowed
    )
    raise SessionCodecError(
        f"Reference {chosen_key!r} has {disc_label}={discriminator!r}, but no allowed "
        f"shape matches (available {disc_label}: {available})"
    )


def _select_allowed_spec_by_fields(
    spec: ReferenceSpec,
    disc_key: str,
    value_fields: object,
) -> CfgSectionSpec | None:
    if not isinstance(value_fields, Iterable):
        return None
    provided = {str(key) for key in value_fields} - {disc_key}
    matches = [
        allowed_spec
        for allowed_spec in spec.allowed
        if set(allowed_spec.fields) - {disc_key} == provided
    ]
    if len(matches) == 1:
        return matches[0]
    return None


__all__ = ["SessionCodecError", "schema_to_raw", "raw_to_schema"]
