"""Pure finished-cfg validation and lowering through narrow runtime ports."""

from __future__ import annotations

import logging
import math
from collections.abc import Collection
from typing import Protocol, TypeAlias

from .model import (
    CenteredSweepSpec,
    CenteredSweepValue,
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
)

logger = logging.getLogger(__name__)

ReferenceKind: TypeAlias = str

_LOCKED_CENTER_ABS_TOL = 1e-12


class ExpressionResolver(Protocol):
    def __call__(self, expr: str, /) -> int | float: ...


class ReferenceResolver(Protocol):
    def __call__(self, kind: ReferenceKind, key: str, /) -> str | None: ...


class RangeFactory(Protocol):
    def __call__(self, start: float, stop: float, /, *, expts: int) -> object: ...


def validate_reference_kinds(
    schema: CfgSchema,
    allowed_kinds: Collection[str],
) -> None:
    """Validate opaque reference kinds against caller-owned application policy."""
    allowed = frozenset(allowed_kinds)
    _validate_reference_kinds_section(schema.spec, allowed, path=[])


def _validate_reference_kinds_section(
    spec: CfgSectionSpec,
    allowed_kinds: frozenset[str],
    *,
    path: list[str],
) -> None:
    for key, node_spec in spec.fields.items():
        node_path = [*path, key]
        if isinstance(node_spec, ReferenceSpec):
            if node_spec.kind not in allowed_kinds:
                allowed = ", ".join(sorted(allowed_kinds))
                raise RuntimeError(
                    f"Config field '{'.'.join(node_path)}' uses unsupported "
                    f"reference kind {node_spec.kind!r}; allowed kinds: {allowed}"
                )
            for allowed_spec in node_spec.allowed:
                _validate_reference_kinds_section(
                    allowed_spec,
                    allowed_kinds,
                    path=node_path,
                )
        elif isinstance(node_spec, CfgSectionSpec):
            _validate_reference_kinds_section(
                node_spec,
                allowed_kinds,
                path=node_path,
            )


def validate_finished_cfg(
    schema: CfgSchema,
    *,
    resolve_reference: ReferenceResolver | None,
) -> None:
    """Validate the static contract at a finished-cfg boundary."""
    _validate_static_section(
        schema.spec,
        schema.value,
        resolve_reference=resolve_reference,
        path=[],
    )


def lower_finished_cfg(
    schema: CfgSchema,
    *,
    resolve_expression: ExpressionResolver | None,
    resolve_reference: ReferenceResolver | None,
    make_range: RangeFactory,
) -> dict[str, object]:
    """Lower a finished cfg while preserving static/dynamic/lower ordering."""
    validate_finished_cfg(schema, resolve_reference=resolve_reference)
    if resolve_expression is not None:
        _validate_dynamic_section(
            schema.spec,
            schema.value,
            resolve_expression=resolve_expression,
            resolve_reference=resolve_reference,
            path=[],
        )
    return _lower_section(
        schema.spec,
        schema.value,
        resolve_expression=resolve_expression,
        resolve_reference=resolve_reference,
        make_range=make_range,
        path=[],
    )


def _coerce_eval_result(value: int | float, type_: type) -> int | float:
    if type_ is float:
        return float(value)
    if type_ is int:
        if not float(value).is_integer():
            raise RuntimeError(f"Expression result {value!r} is not an integer")
        return int(value)
    raise RuntimeError(f"Eval mode only supports int or float, got {type_!r}")


def _resolve_eval(
    value: EvalValue,
    resolve_expression: ExpressionResolver | None,
    *,
    path: str,
    label: str,
    type_: type = float,
) -> int | float:
    if value.resolved is not None:
        resolved = value.resolved
        # A committed snapshot remains authoritative, but drift must stay visible.
        if resolve_expression is not None:
            try:
                fresh = resolve_expression(value.expr)
            except Exception:
                fresh = None
            if fresh is not None and isinstance(fresh, (int, float)):
                if _coerce_eval_result(fresh, type_) != _coerce_eval_result(
                    resolved, type_
                ):
                    logger.warning(
                        "Config field '%s' (%s): EvalValue %r snapshot %r differs "
                        "from current md evaluation %r; using snapshot",
                        path,
                        label,
                        value.expr,
                        resolved,
                        fresh,
                    )
    elif resolve_expression is not None:
        try:
            resolved = resolve_expression(value.expr)
        except Exception as exc:
            raise RuntimeError(
                f"Config field '{path}' ({label}) expression {value.expr!r} "
                f"failed to evaluate: {exc}"
            ) from exc
    else:
        raise RuntimeError(
            f"Config field '{path}' ({label}) expression {value.expr!r} is unresolved"
        )
    if not isinstance(resolved, (int, float)):
        raise RuntimeError(
            f"Config field '{path}' ({label}) resolved to non-numeric value"
        )
    return _coerce_eval_result(resolved, type_)


def _resolve_sweep_edge(
    value: object,
    resolve_expression: ExpressionResolver | None,
    *,
    path: str,
    label: str,
) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, EvalValue):
        return float(
            _resolve_eval(
                value,
                resolve_expression,
                path=path,
                label=label,
                type_=float,
            )
        )
    raise RuntimeError(f"Config field '{path}' ({label}) must be numeric")


def _static_center_value(value: object, *, path: str) -> float | None:
    raw = value.resolved if isinstance(value, EvalValue) else value
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RuntimeError(
            f"Config field '{path}.center' (Sweep center) must be numeric"
        )
    center = float(raw)
    if not math.isfinite(center):
        raise RuntimeError(
            f"Config field '{path}.center' (Sweep center) must be finite"
        )
    return center


def _validate_centered_sweep_contract(
    spec: CenteredSweepSpec,
    value: CenteredSweepValue,
    full_path: str,
    *,
    center: float | None = None,
) -> None:
    if value.expts > 1 and value.span <= 0.0:
        raise RuntimeError(
            f"Config field '{full_path}' ({spec.label}) centered sweep span must be "
            "greater than 0 when expts > 1"
        )
    if spec.locked_center is None:
        return
    center_value = (
        _static_center_value(value.center, path=full_path)
        if center is None
        else float(center)
    )
    if center_value is None:
        return
    if not math.isclose(
        center_value,
        float(spec.locked_center),
        rel_tol=0.0,
        abs_tol=_LOCKED_CENTER_ABS_TOL,
    ):
        raise RuntimeError(
            f"Config field '{full_path}.center' (Sweep center) is locked to "
            f"{float(spec.locked_center)!r}, got {center_value!r}"
        )


def _select_reference_spec(
    ref_spec: ReferenceSpec,
    ref_value: ReferenceValue,
    resolve_reference: ReferenceResolver | None,
) -> CfgSectionSpec:
    chosen = ref_value.chosen_key
    if chosen.startswith("<Custom:"):
        if not chosen.endswith(">"):
            raise RuntimeError(f"Invalid custom reference key: {chosen!r}")
        label = chosen[len("<Custom:") : -1]
        for spec in ref_spec.allowed:
            if spec.label == label:
                return spec
        allowed = ", ".join(spec.label for spec in ref_spec.allowed)
        raise RuntimeError(
            f"Unknown custom reference label {label!r}; allowed labels: {allowed}"
        )

    if resolve_reference is None:
        raise RuntimeError(
            f"Cannot resolve library reference {chosen!r} without ModuleLibrary"
        )

    label = resolve_reference(ref_spec.kind, chosen)
    if label is None:
        raise RuntimeError(f"Unknown {ref_spec.kind} reference: {chosen!r}")

    for spec in ref_spec.allowed:
        if spec.label == label:
            return spec
    allowed = ", ".join(spec.label for spec in ref_spec.allowed)
    raise RuntimeError(
        f"Library reference {chosen!r} resolved to unsupported spec "
        f"{label!r}; allowed labels: {allowed}"
    )


def _lower_section(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    *,
    resolve_expression: ExpressionResolver | None,
    resolve_reference: ReferenceResolver | None,
    make_range: RangeFactory,
    path: list[str],
) -> dict[str, object]:
    result: dict[str, object] = {}
    extra_keys = set(value.fields) - set(spec.fields)
    if extra_keys:
        section = ".".join(path) or "<root>"
        extras = ", ".join(sorted(extra_keys))
        raise RuntimeError(f"Config section '{section}' has unknown fields: {extras}")

    for key, node_spec in spec.fields.items():
        node_value = value.fields.get(key)
        if node_value is None:
            if isinstance(node_spec, LiteralSpec):
                result[key] = node_spec.value
                continue
            if isinstance(node_spec, ReferenceSpec) and node_spec.optional:
                continue
            label = getattr(node_spec, "label", "") or key
            full_path = ".".join([*path, key])
            raise RuntimeError(f"Config field '{full_path}' ({label}) is missing")

        if isinstance(node_spec, ScalarSpec):
            assert isinstance(node_value, (DirectValue, EvalValue))
            if isinstance(node_value, DirectValue):
                if node_value.value is None:
                    if node_spec.optional:
                        continue
                    label = node_spec.label or key
                    full_path = ".".join([*path, key])
                    raise RuntimeError(f"Config field '{full_path}' ({label}) is unset")
                result[key] = node_value.value
            else:
                label = node_spec.label or key
                full_path = ".".join([*path, key])
                result[key] = _resolve_eval(
                    node_value,
                    resolve_expression,
                    path=full_path,
                    label=label,
                    type_=node_spec.type,
                )

        elif isinstance(node_spec, LiteralSpec):
            result[key] = node_spec.value

        elif isinstance(node_spec, SweepSpec):
            assert isinstance(node_value, SweepValue)
            start = _resolve_sweep_edge(
                node_value.start,
                resolve_expression,
                path=".".join([*path, key, "start"]),
                label="Sweep start",
            )
            stop = _resolve_sweep_edge(
                node_value.stop,
                resolve_expression,
                path=".".join([*path, key, "stop"]),
                label="Sweep stop",
            )
            result[key] = make_range(start, stop, expts=node_value.expts)

        elif isinstance(node_spec, CenteredSweepSpec):
            assert isinstance(node_value, CenteredSweepValue)
            full_path = ".".join([*path, key])
            center = _resolve_sweep_edge(
                node_value.center,
                resolve_expression,
                path=f"{full_path}.center",
                label="Sweep center",
            )
            _validate_centered_sweep_contract(
                node_spec, node_value, full_path, center=center
            )
            if node_value.expts == 1:
                result[key] = make_range(center, center, expts=1)
                continue
            half_span = float(node_value.span) / 2.0
            result[key] = make_range(
                center - half_span,
                center + half_span,
                expts=node_value.expts,
            )

        elif isinstance(node_spec, ReferenceSpec):
            assert isinstance(node_value, ReferenceValue)
            if not isinstance(node_value.value, CfgSectionValue):
                label = node_spec.label or key
                full_path = ".".join([*path, key])
                raise RuntimeError(f"Config field '{full_path}' ({label}) is missing")
            result[key] = _lower_section(
                _select_reference_spec(node_spec, node_value, resolve_reference),
                node_value.value,
                resolve_expression=resolve_expression,
                resolve_reference=resolve_reference,
                make_range=make_range,
                path=[*path, key],
            )

        elif isinstance(node_spec, DeviceRefSpec):
            assert isinstance(node_value, DirectValue)
            if not isinstance(node_value.value, str) or not node_value.value:
                label = node_spec.label or key
                full_path = ".".join([*path, key])
                raise RuntimeError(f"Config field '{full_path}' ({label}) is unset")
            result[key] = node_value.value

        elif isinstance(node_spec, CfgSectionSpec):
            assert isinstance(node_value, CfgSectionValue)
            result[key] = _lower_section(
                node_spec,
                node_value,
                resolve_expression=resolve_expression,
                resolve_reference=resolve_reference,
                make_range=make_range,
                path=[*path, key],
            )

        else:
            raise TypeError(f"Unknown CfgNodeSpec type: {type(node_spec)}")

    return result


def _validate_static_section(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    *,
    resolve_reference: ReferenceResolver | None,
    path: list[str],
) -> None:
    extra = set(value.fields) - set(spec.fields)
    if extra:
        section = ".".join(path) or "<root>"
        raise RuntimeError(
            f"Config section '{section}' has unknown fields: {', '.join(sorted(extra))}"
        )
    for key, node_spec in spec.fields.items():
        full_path = ".".join([*path, key])
        if key not in value.fields:
            raise RuntimeError(f"Config field '{full_path}' is missing from the value")
        _validate_static_node(
            node_spec,
            value.fields[key],
            resolve_reference=resolve_reference,
            full_path=full_path,
        )


def _validate_static_node(
    spec: object,
    node_value: object,
    *,
    resolve_reference: ReferenceResolver | None,
    full_path: str,
) -> None:
    if node_value is None:
        if isinstance(spec, ReferenceSpec) and spec.optional:
            return
        raise RuntimeError(
            f"Config field '{full_path}' is None but is not a disabled optional ref"
        )

    if isinstance(spec, LiteralSpec):
        if not isinstance(node_value, DirectValue) or node_value.value != spec.value:
            raise RuntimeError(
                f"Config field '{full_path}' is a locked literal "
                f"(must be {spec.value!r}), got {node_value!r}"
            )
        return

    if isinstance(spec, ScalarSpec):
        if isinstance(node_value, EvalValue):
            return
        if not isinstance(node_value, DirectValue):
            raise RuntimeError(
                f"Config field '{full_path}' must be a DirectValue/EvalValue, "
                f"got {type(node_value).__name__}"
            )
        _validate_scalar(spec, node_value, full_path)
        return

    if isinstance(spec, SweepSpec):
        if not isinstance(node_value, SweepValue):
            raise RuntimeError(
                f"Config field '{full_path}' must be a SweepValue, "
                f"got {type(node_value).__name__}"
            )
        return

    if isinstance(spec, CenteredSweepSpec):
        if not isinstance(node_value, CenteredSweepValue):
            raise RuntimeError(
                f"Config field '{full_path}' must be a CenteredSweepValue, "
                f"got {type(node_value).__name__}"
            )
        _validate_centered_sweep_contract(spec, node_value, full_path)
        return

    if isinstance(spec, DeviceRefSpec):
        if not isinstance(node_value, DirectValue):
            raise RuntimeError(
                f"Config field '{full_path}' (device ref) must be a DirectValue, "
                f"got {type(node_value).__name__}"
            )
        return

    if isinstance(spec, ReferenceSpec):
        if not isinstance(node_value, ReferenceValue):
            raise RuntimeError(
                f"Config field '{full_path}' must be an enabled module/waveform "
                f"ref, got {type(node_value).__name__}"
            )
        chosen_spec = _select_reference_spec(spec, node_value, resolve_reference)
        _validate_static_section(
            chosen_spec,
            node_value.value,
            resolve_reference=resolve_reference,
            path=[full_path],
        )
        return

    if isinstance(spec, CfgSectionSpec):
        if not isinstance(node_value, CfgSectionValue):
            raise RuntimeError(
                f"Config field '{full_path}' must be a CfgSectionValue, "
                f"got {type(node_value).__name__}"
            )
        _validate_static_section(
            spec,
            node_value,
            resolve_reference=resolve_reference,
            path=[full_path],
        )
        return

    raise RuntimeError(
        f"Config field '{full_path}': unknown spec {type(spec).__name__}"
    )


def _validate_scalar(spec: ScalarSpec, node_value: DirectValue, full_path: str) -> None:
    value = node_value.value
    if value is None:
        return
    if spec.type is bool:
        valid = isinstance(value, bool)
    elif spec.type is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif spec.type is float:
        valid = isinstance(value, float) or (
            isinstance(value, int) and not isinstance(value, bool)
        )
    elif spec.type is str:
        valid = isinstance(value, str)
    else:
        valid = isinstance(value, spec.type)
    if not valid:
        if isinstance(value, str) and spec.type in (int, float, bool):
            raise RuntimeError(
                f"Config field '{full_path}' received string {value!r} where a "
                f"{spec.type.__name__} was expected (the numeric value was not "
                f"coerced — it arrived as a string)"
            )
        raise RuntimeError(
            f"Config field '{full_path}' value {value!r} is not compatible with "
            f"spec type {spec.type.__name__}"
        )
    if spec.choices is not None and value not in spec.choices:
        raise RuntimeError(
            f"Config field '{full_path}' value {value!r} is not in allowed choices "
            f"{spec.choices!r}"
        )


def _validate_dynamic_section(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    *,
    resolve_expression: ExpressionResolver,
    resolve_reference: ReferenceResolver | None,
    path: list[str],
) -> None:
    for key, node_spec in spec.fields.items():
        full_path = ".".join([*path, key])
        _validate_dynamic_node(
            node_spec,
            value.fields.get(key),
            resolve_expression=resolve_expression,
            resolve_reference=resolve_reference,
            full_path=full_path,
        )


def _validate_dynamic_node(
    spec: object,
    node_value: object,
    *,
    resolve_expression: ExpressionResolver,
    resolve_reference: ReferenceResolver | None,
    full_path: str,
) -> None:
    if node_value is None:
        return

    if isinstance(spec, LiteralSpec):
        return

    if isinstance(spec, ScalarSpec):
        if isinstance(node_value, DirectValue):
            if node_value.value is None:
                if spec.optional:
                    return
                label = spec.label or full_path.rsplit(".", 1)[-1]
                raise RuntimeError(
                    f"Config field '{full_path}' ({label}) is unset (no value to lower)"
                )
            return
        if isinstance(node_value, EvalValue):
            label = spec.label or full_path.rsplit(".", 1)[-1]
            _validate_eval(
                node_value,
                resolve_expression,
                spec.type,
                full_path,
                label,
            )
        return

    if isinstance(spec, SweepSpec):
        if isinstance(node_value, SweepValue):
            if isinstance(node_value.start, EvalValue):
                _validate_eval(
                    node_value.start,
                    resolve_expression,
                    float,
                    f"{full_path}.start",
                    "Sweep start",
                )
            if isinstance(node_value.stop, EvalValue):
                _validate_eval(
                    node_value.stop,
                    resolve_expression,
                    float,
                    f"{full_path}.stop",
                    "Sweep stop",
                )
        return

    if isinstance(spec, CenteredSweepSpec):
        if isinstance(node_value, CenteredSweepValue):
            _validate_centered_sweep_contract(spec, node_value, full_path)
            if isinstance(node_value.center, EvalValue):
                _validate_eval(
                    node_value.center,
                    resolve_expression,
                    float,
                    f"{full_path}.center",
                    "Sweep center",
                )
            if spec.locked_center is not None:
                center = _resolve_sweep_edge(
                    node_value.center,
                    resolve_expression,
                    path=f"{full_path}.center",
                    label="Sweep center",
                )
                _validate_centered_sweep_contract(
                    spec, node_value, full_path, center=center
                )
        return

    if isinstance(spec, DeviceRefSpec):
        if isinstance(node_value, DirectValue):
            if not isinstance(node_value.value, str) or not node_value.value:
                label = spec.label or full_path.rsplit(".", 1)[-1]
                raise RuntimeError(
                    f"Config field '{full_path}' ({label}) device not selected"
                )
        return

    if isinstance(spec, ReferenceSpec):
        if isinstance(node_value, ReferenceValue):
            if isinstance(node_value.value, CfgSectionValue):
                chosen_spec = _select_reference_spec(
                    spec, node_value, resolve_reference
                )
                _validate_dynamic_section(
                    chosen_spec,
                    node_value.value,
                    resolve_expression=resolve_expression,
                    resolve_reference=resolve_reference,
                    path=[full_path],
                )
        return

    if isinstance(spec, CfgSectionSpec):
        if isinstance(node_value, CfgSectionValue):
            _validate_dynamic_section(
                spec,
                node_value,
                resolve_expression=resolve_expression,
                resolve_reference=resolve_reference,
                path=[full_path],
            )


def _validate_eval(
    value: EvalValue,
    resolve_expression: ExpressionResolver,
    type_: type,
    full_path: str,
    label: str,
) -> None:
    try:
        result = resolve_expression(value.expr)
        _coerce_eval_result(result, type_)
    except Exception as exc:
        raise RuntimeError(
            f"Config field '{full_path}' ({label}) expression "
            f"{value.expr!r} failed: {exc}"
        ) from exc
