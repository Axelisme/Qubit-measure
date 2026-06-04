"""Session cfg codec — pure raw↔live transforms for tab cfg persistence.

WorkspaceService's internal implementation of capturing/restoring a tab's cfg:
``schema_to_raw`` lowers a live ``CfgSchema`` to a JSON-able dict (``cfg_raw`` in
the memento); ``raw_to_schema`` rebuilds the live value tree from that dict given
the adapter's base spec. Pure functions, no I/O, no state — the persisted
``cfg_raw`` is opaque to the Caretaker; only this codec knows its shape.

Moved out of the former ``SessionPersistenceService`` (Phase 126): the disk I/O
went to the Caretaker, this codec stays as WorkspaceService's helper.
"""

from __future__ import annotations

from typing import Optional, Union

from zcu_tools.gui.app.main.adapter import (
    CfgNodeSpec,
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    DisabledRefValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    make_default_value,
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
    spec: CfgSectionSpec, value: CfgSectionValue
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, node_spec in spec.fields.items():
        node_val = value.fields.get(key)
        if node_val is None:
            continue
        payload[key] = _node_value_to_raw(node_spec, node_val)
    return payload


def _node_value_to_raw(spec: CfgNodeSpec, value: CfgNodeValue) -> object:
    if isinstance(spec, LiteralSpec):
        # Fixed-value field; the literal value is canonical.
        return {
            "__kind": "direct",
            "value": _to_json_compatible(spec.value),
            "is_unset": False,
        }
    if isinstance(spec, ScalarSpec):
        assert isinstance(value, (DirectValue, EvalValue))
        if isinstance(value, EvalValue):
            return {
                "__kind": "eval",
                "expr": value.expr,
            }
        return {
            "__kind": "direct",
            "value": _to_json_compatible(value.value),
            "is_unset": value.is_unset,
        }
    if isinstance(spec, SweepSpec):
        assert isinstance(value, SweepValue)
        return {
            "start": _sweep_edge_to_raw(value.start),
            "stop": _sweep_edge_to_raw(value.stop),
            "expts": value.expts,
            "step": value.step,
        }
    if isinstance(spec, DeviceRefSpec):
        assert isinstance(value, DirectValue)
        return {
            "__kind": "direct",
            "value": _to_json_compatible(value.value),
            "is_unset": value.is_unset,
        }
    if isinstance(spec, CfgSectionSpec):
        assert isinstance(value, CfgSectionValue)
        return _section_value_to_raw(spec, value)
    if isinstance(spec, ModuleRefSpec):
        # An optional ModuleRef that is disabled carries a DisabledRefValue
        # marker (ADR-0012), not a ModuleRefValue — persist it as such so the
        # disabled state survives reload.
        if isinstance(value, DisabledRefValue):
            return {"__kind": "disabled"}
        assert isinstance(value, ModuleRefValue)
        return {
            "__kind": "module_ref",
            "chosen_key": value.chosen_key,
            "is_overridden": value.is_overridden,
            "value": _section_value_to_raw(
                _select_allowed_spec_for_restore(spec, value.chosen_key),
                value.value,
            ),
        }
    if isinstance(spec, WaveformRefSpec):
        if isinstance(value, DisabledRefValue):
            return {"__kind": "disabled"}
        assert isinstance(value, WaveformRefValue)
        return {
            "__kind": "waveform_ref",
            "chosen_key": value.chosen_key,
            "is_overridden": value.is_overridden,
            "value": _section_value_to_raw(
                _select_allowed_spec_for_restore(spec, value.chosen_key),
                value.value,
            ),
        }
    return _to_json_compatible(value)


def _sweep_edge_to_raw(value: Union[float, EvalValue]) -> object:
    if isinstance(value, EvalValue):
        return {"__kind": "eval", "expr": value.expr}
    return float(value)


def _section_value_from_raw(
    spec: CfgSectionSpec,
    raw: dict[str, object],
) -> CfgSectionValue:
    value = make_default_value(spec)
    for key, node_spec in spec.fields.items():
        if key not in raw:
            continue
        parsed = _node_value_from_raw(node_spec, raw[key])
        if parsed is not None:
            value.fields[key] = parsed
    return value


def _node_value_from_raw(
    spec: CfgNodeSpec,
    raw: object,
) -> Optional[CfgNodeValue]:
    if isinstance(spec, ScalarSpec):
        if (
            isinstance(raw, dict)
            and raw.get("__kind") == "eval"
            and isinstance(raw.get("expr"), str)
        ):
            return EvalValue(expr=raw["expr"], resolved=None, error=None)
        if isinstance(raw, dict) and raw.get("__kind") == "direct":
            value = raw.get("value")
            is_unset = bool(raw.get("is_unset", False))
            return DirectValue(value=value, is_unset=is_unset)
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
    if isinstance(spec, DeviceRefSpec):
        if isinstance(raw, dict) and raw.get("__kind") == "direct":
            value = raw.get("value")
            if not isinstance(value, str):
                raise RuntimeError("Device reference value must be string")
            return DirectValue(value, is_unset=bool(raw.get("is_unset", False)))
        raise RuntimeError("Device reference must use direct payload encoding")
    if isinstance(spec, CfgSectionSpec):
        if not isinstance(raw, dict):
            raise RuntimeError("Section payload must be an object")
        return _section_value_from_raw(spec, raw)
    if isinstance(spec, ModuleRefSpec):
        return _ref_value_from_raw(spec, raw)
    if isinstance(spec, WaveformRefSpec):
        return _waveform_ref_value_from_raw(spec, raw)
    return None


def _parse_sweep_edge(raw: object) -> Union[float, EvalValue]:
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
    spec: ModuleRefSpec,
    raw: object,
) -> Union[ModuleRefValue, DisabledRefValue]:
    if isinstance(raw, dict) and raw.get("__kind") == "disabled":
        return DisabledRefValue()
    if (
        isinstance(raw, dict)
        and raw.get("__kind") == "module_ref"
        and isinstance(raw.get("chosen_key"), str)
        and isinstance(raw.get("value"), dict)
    ):
        chosen_key = raw["chosen_key"]
        value_spec = _select_allowed_spec_for_restore(spec, chosen_key)
        nested = _section_value_from_raw(value_spec, raw["value"])
        return ModuleRefValue(
            chosen_key=chosen_key,
            value=nested,
            is_overridden=bool(raw.get("is_overridden", False)),
        )
    raise RuntimeError("Module reference must use module_ref payload encoding")


def _waveform_ref_value_from_raw(
    spec: WaveformRefSpec,
    raw: object,
) -> Union[WaveformRefValue, DisabledRefValue]:
    if isinstance(raw, dict) and raw.get("__kind") == "disabled":
        return DisabledRefValue()
    if (
        isinstance(raw, dict)
        and raw.get("__kind") == "waveform_ref"
        and isinstance(raw.get("chosen_key"), str)
        and isinstance(raw.get("value"), dict)
    ):
        chosen_key = raw["chosen_key"]
        value_spec = _select_allowed_spec_for_restore(spec, chosen_key)
        nested = _section_value_from_raw(value_spec, raw["value"])
        return WaveformRefValue(
            chosen_key=chosen_key,
            value=nested,
            is_overridden=bool(raw.get("is_overridden", False)),
        )
    raise RuntimeError("Waveform reference must use waveform_ref payload encoding")


def _select_allowed_spec_for_restore(
    spec: Union[ModuleRefSpec, WaveformRefSpec], chosen_key: str
) -> CfgSectionSpec:
    if chosen_key.startswith("<Custom:") and chosen_key.endswith(">"):
        label = chosen_key[len("<Custom:") : -1]
        for allowed_spec in spec.allowed:
            if allowed_spec.label == label:
                return allowed_spec
    return spec.allowed[0]


__all__ = ["SessionCodecError", "schema_to_raw", "raw_to_schema"]
