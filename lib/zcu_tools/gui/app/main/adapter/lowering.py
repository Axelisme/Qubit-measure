from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from .types import (
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
)

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _resolve_eval(
    value: EvalValue,
    md: MetaDict | None,
    *,
    path: str,
    label: str,
    type_: type = float,
) -> int | float:
    """Resolve an EvalValue to a number, coerced to ``type_``.

    Prefers the snapshot ``value.resolved``; when absent (adapters may build an
    EvalValue without one), evaluates ``value.expr`` against ``md`` — lowering is
    the single place that owns resolution. Fails only if there is no snapshot and
    no md to evaluate against, or the expression itself is invalid.

    ``type_`` is the owning ScalarSpec's physical type; the result is coerced via
    ``coerce_eval_result`` (int spec → int, float spec → float) so that e.g. an
    ``EvalValue("ro_ch")`` for an int channel lowers to ``int`` rather than float.
    """
    from zcu_tools.gui.app.main.expression import coerce_eval_result

    if value.resolved is not None:
        resolved = value.resolved
        # The snapshot wins, but if md is available cross-check it against a
        # fresh evaluation: a mismatch means the snapshot is stale (md changed
        # after the field was set). Log it — never silently lower a stale value
        # without trace — but keep the snapshot to preserve existing semantics.
        if md is not None:
            from zcu_tools.gui.app.main.expression import evaluate_numeric_expr

            try:
                fresh = evaluate_numeric_expr(value.expr, md)
            except Exception:
                fresh = None
            # Compare coerced-to-spec-type so an int/float representation
            # difference (5 vs 5.0) is not flagged as a drift.
            if fresh is not None and isinstance(fresh, (int, float)):
                if coerce_eval_result(fresh, type_) != coerce_eval_result(
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
    elif md is not None:
        from zcu_tools.gui.app.main.expression import evaluate_numeric_expr

        try:
            resolved = evaluate_numeric_expr(value.expr, md)
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
    return coerce_eval_result(resolved, type_)


def _resolve_sweep_edge(
    value: object, md: MetaDict | None, *, path: str, label: str
) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, EvalValue):
        # Sweep edges have no per-edge ScalarSpec; scan values are always float.
        return float(_resolve_eval(value, md, path=path, label=label, type_=float))
    raise RuntimeError(f"Config field '{path}' ({label}) must be numeric")


def _section_to_dict_inner(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    ml: ModuleLibrary | None,
    path: list[str],
    md: MetaDict | None = None,
) -> dict:
    result: dict[str, Any] = {}
    extra_keys = set(value.fields.keys()) - set(spec.fields.keys())
    if extra_keys:
        section = ".".join(path) or "<root>"
        extras = ", ".join(sorted(extra_keys))
        raise RuntimeError(f"Config section '{section}' has unknown fields: {extras}")
    for key, node_spec in spec.fields.items():
        node_val = value.fields.get(key)
        # ``None`` is a disabled optional ref (ADR-0010) — lowering omits it (the
        # exp cfg has no such field). This is the *only* place "disabled →
        # disappears": run/save is the boundary where an unused field drops out.
        if node_val is None:
            if isinstance(node_spec, LiteralSpec):
                result[key] = node_spec.value
                continue
            if (
                isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec))
                and node_spec.optional
            ):
                continue  # disabled optional ModuleRef/WaveformRef → omit from output
            label = getattr(node_spec, "label", "") or key
            full_path = ".".join([*path, key])
            raise RuntimeError(f"Config field '{full_path}' ({label}) is missing")

        if isinstance(node_spec, ScalarSpec):
            assert isinstance(node_val, (DirectValue, EvalValue))
            if isinstance(node_val, DirectValue):
                if node_val.value is None:
                    # An optional unset scalar is omitted from the lowered cfg so
                    # the model default (typically None) applies; a non-optional
                    # unset scalar is an error (must be filled).
                    if node_spec.optional:
                        continue
                    label = node_spec.label or key
                    full_path = ".".join([*path, key])
                    raise RuntimeError(f"Config field '{full_path}' ({label}) is unset")
                result[key] = node_val.value
            else:
                label = node_spec.label or key
                full_path = ".".join([*path, key])
                result[key] = _resolve_eval(
                    node_val, md, path=full_path, label=label, type_=node_spec.type
                )

        elif isinstance(node_spec, LiteralSpec):
            result[key] = node_spec.value

        elif isinstance(node_spec, SweepSpec):
            assert isinstance(node_val, SweepValue)
            from zcu_tools.notebook.utils import make_sweep

            start = _resolve_sweep_edge(
                node_val.start,
                md,
                path=".".join([*path, key, "start"]),
                label="Sweep start",
            )
            stop = _resolve_sweep_edge(
                node_val.stop,
                md,
                path=".".join([*path, key, "stop"]),
                label="Sweep stop",
            )
            result[key] = make_sweep(start, stop, expts=node_val.expts)

        elif isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec)):
            assert isinstance(node_val, (ModuleRefValue, WaveformRefValue))
            if not isinstance(node_val.value, CfgSectionValue):
                label = node_spec.label or key
                full_path = ".".join([*path, key])
                raise RuntimeError(f"Config field '{full_path}' ({label}) is missing")
            result[key] = _section_to_dict_inner(
                find_allowed_spec(node_spec, node_val, ml),
                node_val.value,
                ml,
                [*path, key],
                md,
            )

        elif isinstance(node_spec, DeviceRefSpec):
            assert isinstance(node_val, DirectValue)
            if not isinstance(node_val.value, str) or not node_val.value:
                label = node_spec.label or key
                full_path = ".".join([*path, key])
                raise RuntimeError(f"Config field '{full_path}' ({label}) is unset")
            result[key] = node_val.value

        elif isinstance(node_spec, CfgSectionSpec):
            assert isinstance(node_val, CfgSectionValue)
            result[key] = _section_to_dict_inner(
                node_spec, node_val, ml, [*path, key], md
            )

        else:
            raise TypeError(f"Unknown CfgNodeSpec type: {type(node_spec)}")

    return result


def find_allowed_spec(
    ref_spec: ModuleRefSpec | WaveformRefSpec,
    ref_val: ModuleRefValue | WaveformRefValue,
    ml: ModuleLibrary | None,
) -> CfgSectionSpec:
    """Return the CfgSectionSpec from allowed[] that matches chosen_key's label."""
    chosen = ref_val.chosen_key
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

    if ml is None:
        raise RuntimeError(
            f"Cannot resolve library reference {chosen!r} without ModuleLibrary"
        )

    from zcu_tools.gui.app.main.cfg_schemas import (
        module_cfg_to_value,
        waveform_cfg_to_value,
    )

    if isinstance(ref_spec, ModuleRefSpec):
        if chosen not in ml.modules:
            raise RuntimeError(f"Unknown module reference: {chosen!r}")
        chosen_spec, _ = module_cfg_to_value(ml.modules[chosen])
    else:
        if chosen not in ml.waveforms:
            raise RuntimeError(f"Unknown waveform reference: {chosen!r}")
        chosen_spec, _ = waveform_cfg_to_value(ml.waveforms[chosen])

    for spec in ref_spec.allowed:
        if spec.label == chosen_spec.label:
            return spec
    allowed = ", ".join(spec.label for spec in ref_spec.allowed)
    raise RuntimeError(
        f"Library reference {chosen!r} resolved to unsupported spec "
        f"{chosen_spec.label!r}; allowed labels: {allowed}"
    )


# ---------------------------------------------------------------------------
# Static schema validation — a value tree must be complete and spec-compliant.
#
# This is the "static" half (structure + scalar type/choices + literal equality)
# that holds regardless of MetaDict — checked at the *finished-cfg* boundaries
# (``make_default_cfg`` output, ``to_raw_dict`` before lowering). The "dynamic"
# half (required must have a value, EvalValue must resolve) is enforced by
# lowering itself with the live md. EvalValue is skipped here — its scalar type
# is only fixed when resolved at lower time.
# ---------------------------------------------------------------------------


def validate_section(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    ml: ModuleLibrary | None,
    path: list[str],
) -> None:
    """Fast-fail if ``value`` is structurally incomplete or violates ``spec``.

    Raises ``RuntimeError`` at the first problem (the path is reported). A value
    tree produced by an adapter must be complete (every spec field has an entry)
    and statically spec-compliant; this is the single static-validation walk.
    """
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
        node_val = value.fields[key]
        _validate_node(node_spec, node_val, ml, full_path)


def _validate_node(
    spec: object,
    node_val: object,
    ml: ModuleLibrary | None,
    full_path: str,
) -> None:
    # A ``None`` entry is a disabled optional ref (ADR-0010); legal only there.
    if node_val is None:
        if isinstance(spec, (ModuleRefSpec, WaveformRefSpec)) and spec.optional:
            return
        raise RuntimeError(
            f"Config field '{full_path}' is None but is not a disabled optional ref"
        )

    if isinstance(spec, LiteralSpec):
        if not isinstance(node_val, DirectValue) or node_val.value != spec.value:
            raise RuntimeError(
                f"Config field '{full_path}' is a locked literal "
                f"(must be {spec.value!r}), got {node_val!r}"
            )
        return

    if isinstance(spec, ScalarSpec):
        # EvalValue's scalar type is only fixed when resolved at lower time — skip.
        if isinstance(node_val, EvalValue):
            return
        if not isinstance(node_val, DirectValue):
            raise RuntimeError(
                f"Config field '{full_path}' must be a DirectValue/EvalValue, "
                f"got {type(node_val).__name__}"
            )
        _validate_scalar(spec, node_val, full_path)
        return

    if isinstance(spec, SweepSpec):
        if not isinstance(node_val, SweepValue):
            raise RuntimeError(
                f"Config field '{full_path}' must be a SweepValue, "
                f"got {type(node_val).__name__}"
            )
        return

    if isinstance(spec, DeviceRefSpec):
        if not isinstance(node_val, DirectValue):
            raise RuntimeError(
                f"Config field '{full_path}' (device ref) must be a DirectValue, "
                f"got {type(node_val).__name__}"
            )
        return

    if isinstance(spec, (ModuleRefSpec, WaveformRefSpec)):
        if not isinstance(node_val, (ModuleRefValue, WaveformRefValue)):
            raise RuntimeError(
                f"Config field '{full_path}' must be an enabled module/waveform "
                f"ref, got {type(node_val).__name__}"
            )
        chosen_spec = find_allowed_spec(spec, node_val, ml)
        validate_section(chosen_spec, node_val.value, ml, [full_path])
        return

    if isinstance(spec, CfgSectionSpec):
        if not isinstance(node_val, CfgSectionValue):
            raise RuntimeError(
                f"Config field '{full_path}' must be a CfgSectionValue, "
                f"got {type(node_val).__name__}"
            )
        validate_section(spec, node_val, ml, [full_path])
        return

    raise RuntimeError(
        f"Config field '{full_path}': unknown spec {type(spec).__name__}"
    )


def _validate_scalar(spec: ScalarSpec, node_val: DirectValue, full_path: str) -> None:
    val = node_val.value
    # A None DirectValue is "unset" — a legal intermediate state; the *dynamic*
    # "required must have a value" check belongs to lowering, not here.
    if val is None:
        return
    # Type check (widen-only): int may stand in for a float field; bool/str/float
    # must match exactly; float must NOT narrow to an int field.
    if spec.type is bool:
        ok = isinstance(val, bool)
    elif spec.type is int:
        ok = isinstance(val, int) and not isinstance(val, bool)
    elif spec.type is float:
        # int widens to float (5 -> 5.0); bool excluded.
        ok = isinstance(val, float) or (
            isinstance(val, int) and not isinstance(val, bool)
        )
    elif spec.type is str:
        ok = isinstance(val, str)
    else:
        ok = isinstance(val, spec.type)
    if not ok:
        # A string standing in for a numeric/bool field almost always means the
        # value arrived un-coerced (e.g. an MCP client stringified a number
        # against a schema's "string" member): the literal is a valid value, it
        # just kept the wrong type. Call that out specifically so the cause is
        # "coercion", not "wrong value".
        if isinstance(val, str) and spec.type in (int, float, bool):
            raise RuntimeError(
                f"Config field '{full_path}' received string {val!r} where a "
                f"{spec.type.__name__} was expected (the numeric value was not "
                f"coerced — it arrived as a string)"
            )
        raise RuntimeError(
            f"Config field '{full_path}' value {val!r} is not compatible with "
            f"spec type {spec.type.__name__}"
        )
    if spec.choices is not None and val not in spec.choices:
        raise RuntimeError(
            f"Config field '{full_path}' value {val!r} is not in allowed choices "
            f"{spec.choices!r}"
        )


# ---------------------------------------------------------------------------
# Dynamic schema validation — every value must be lowerable with the given md.
#
# This is the "dynamic" half: every scalar must have a value (no
# DirectValue(None)), every EvalValue must resolve against md, every device
# ref must be selected. Checked by ``to_raw_dict`` before lowering when md
# is available. The lowering itself has its own (overlapping) checks as a
# safety net (plan A — redundant but safe).
# ---------------------------------------------------------------------------


def validate_dynamic_section(
    spec: CfgSectionSpec,
    value: CfgSectionValue,
    md: MetaDict,
    ml: ModuleLibrary | None,
    path: list[str],
) -> None:
    """Fast-fail if ``value`` cannot be lowered with the given ``md``.

    Raises ``RuntimeError`` at the first problem (the path is reported).
    """
    for key, node_spec in spec.fields.items():
        full_path = ".".join([*path, key])
        node_val = value.fields.get(key)
        _validate_dynamic_node(node_spec, node_val, md, ml, full_path)


def _validate_dynamic_node(
    spec: object,
    node_val: object,
    md: MetaDict,
    ml: ModuleLibrary | None,
    full_path: str,
) -> None:
    if node_val is None:
        if isinstance(spec, (ModuleRefSpec, WaveformRefSpec)) and spec.optional:
            return
        # Non-optional None is a static error (caught by validate_section);
        # if we reach here it means static validation was skipped — bail out.
        return

    if isinstance(spec, LiteralSpec):
        return

    if isinstance(spec, ScalarSpec):
        if isinstance(node_val, DirectValue):
            if node_val.value is None:
                # Optional unset scalar lowers to an omitted key (model default) —
                # not an error. Non-optional unset has no value to lower.
                if spec.optional:
                    return
                label = spec.label or full_path.rsplit(".", 1)[-1]
                raise RuntimeError(
                    f"Config field '{full_path}' ({label}) is unset (no value to lower)"
                )
            return
        if isinstance(node_val, EvalValue):
            label = spec.label or full_path.rsplit(".", 1)[-1]
            _validate_eval(node_val, md, spec.type, full_path, label)
            return
        return

    if isinstance(spec, SweepSpec):
        if isinstance(node_val, SweepValue):
            if isinstance(node_val.start, EvalValue):
                _validate_eval(
                    node_val.start, md, float, f"{full_path}.start", "Sweep start"
                )
            if isinstance(node_val.stop, EvalValue):
                _validate_eval(
                    node_val.stop, md, float, f"{full_path}.stop", "Sweep stop"
                )
        return

    if isinstance(spec, DeviceRefSpec):
        if isinstance(node_val, DirectValue):
            if not isinstance(node_val.value, str) or not node_val.value:
                label = spec.label or full_path.rsplit(".", 1)[-1]
                raise RuntimeError(
                    f"Config field '{full_path}' ({label}) device not selected"
                )
        return

    if isinstance(spec, (ModuleRefSpec, WaveformRefSpec)):
        if isinstance(node_val, (ModuleRefValue, WaveformRefValue)):
            if isinstance(node_val.value, CfgSectionValue):
                chosen_spec = find_allowed_spec(spec, node_val, ml)
                validate_dynamic_section(
                    chosen_spec, node_val.value, md, ml, [full_path]
                )
        return

    if isinstance(spec, CfgSectionSpec):
        if isinstance(node_val, CfgSectionValue):
            validate_dynamic_section(spec, node_val, md, ml, [full_path])
        return


def _validate_eval(
    value: EvalValue,
    md: MetaDict,
    type_: type,
    full_path: str,
    label: str,
) -> None:
    from zcu_tools.gui.app.main.expression import (
        coerce_eval_result,
        evaluate_numeric_expr,
    )

    try:
        result = evaluate_numeric_expr(value.expr, md)
        coerce_eval_result(result, type_)
    except Exception as exc:
        raise RuntimeError(
            f"Config field '{full_path}' ({label}) expression "
            f"{value.expr!r} failed: {exc}"
        ) from exc
