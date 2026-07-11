"""Domain-free raw-to-value traversal for an already selected cfg spec."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Never, Protocol

from .inheritance import select_ref_value_spec
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
    ScalarValue,
    SweepSpec,
    SweepValue,
)


@dataclass(frozen=True, slots=True)
class RawMissing:
    """Nominal sentinel type for a key absent from the raw mapping."""


RAW_MISSING = RawMissing()


@dataclass(frozen=True, slots=True)
class ReferenceMaterialization:
    """A policy-selected reference shape, source payload, and persisted key."""

    spec: CfgSectionSpec
    raw: object | RawMissing
    chosen_key: str


class SpecMaterializationPolicy(Protocol):
    """Ports for semantics that cannot be inferred from a generic cfg spec."""

    def scalar_value(
        self,
        path: tuple[str, ...],
        spec: ScalarSpec,
        raw: object | RawMissing,
    ) -> ScalarValue: ...

    def sweep_value(
        self,
        path: tuple[str, ...],
        spec: SweepSpec,
        raw: object | RawMissing,
    ) -> SweepValue: ...

    def centered_sweep_value(
        self,
        path: tuple[str, ...],
        spec: CenteredSweepSpec,
        raw: object | RawMissing,
    ) -> CenteredSweepValue: ...

    def missing_section_value(
        self,
        path: tuple[str, ...],
        spec: CfgSectionSpec,
        raw: object | RawMissing,
    ) -> CfgSectionValue: ...

    def reference_value(
        self,
        path: tuple[str, ...],
        spec: ReferenceSpec,
        raw: object | RawMissing,
    ) -> ReferenceMaterialization | None: ...


def materialize_spec_value(
    spec: CfgSectionSpec,
    raw: Mapping[str, object],
    *,
    policy: SpecMaterializationPolicy,
) -> CfgSectionValue:
    """Walk ``spec`` and build its complete value tree from ``raw``.

    The walker owns only Spec/Value mechanics. Missing scalar, missing/non-mapping
    section, reference selection, and sweep carriers are delegated to ``policy``.
    Extra raw keys are deliberately ignored because a Spec may expose an
    intentional editable subset of a larger domain object.
    """

    return _materialize_section(spec, raw, policy=policy, path=())


def _materialize_section(
    spec: CfgSectionSpec,
    raw: Mapping[str, object],
    *,
    policy: SpecMaterializationPolicy,
    path: tuple[str, ...],
) -> CfgSectionValue:
    fields: dict[str, CfgNodeValue | None] = {}
    for key, node_spec in spec.fields.items():
        node_path = (*path, key)
        raw_value: object | RawMissing = raw.get(key, RAW_MISSING)
        fields[key] = _materialize_node(
            node_spec,
            raw_value,
            policy=policy,
            path=node_path,
        )
    return CfgSectionValue(fields=fields)


def _materialize_node(
    spec: CfgNodeSpec,
    raw: object | RawMissing,
    *,
    policy: SpecMaterializationPolicy,
    path: tuple[str, ...],
) -> CfgNodeValue | None:
    if isinstance(spec, LiteralSpec):
        return DirectValue(spec.value)
    if isinstance(spec, ScalarSpec):
        return policy.scalar_value(path, spec, raw)
    if isinstance(spec, SweepSpec):
        return policy.sweep_value(path, spec, raw)
    if isinstance(spec, CenteredSweepSpec):
        return policy.centered_sweep_value(path, spec, raw)
    if isinstance(spec, ReferenceSpec):
        selected = policy.reference_value(path, spec, raw)
        if selected is None:
            if spec.optional:
                return None
            raise RuntimeError(f"Required reference {'.'.join(path)!r} is missing")
        if not any(selected.spec is allowed for allowed in spec.allowed):
            raise RuntimeError(
                f"Reference policy selected a spec outside allowed shapes at "
                f"{'.'.join(path)!r}"
            )
        if isinstance(selected.raw, Mapping):
            value = _materialize_section(
                selected.spec,
                selected.raw,
                policy=policy,
                path=path,
            )
        else:
            value = policy.missing_section_value(path, selected.spec, selected.raw)
            _validate_section_value(selected.spec, value, path=path)
        return ReferenceValue(chosen_key=selected.chosen_key, value=value)
    if isinstance(spec, CfgSectionSpec):
        if isinstance(raw, Mapping):
            return _materialize_section(spec, raw, policy=policy, path=path)
        value = policy.missing_section_value(path, spec, raw)
        _validate_section_value(spec, value, path=path)
        return value
    raise TypeError(
        f"Unsupported cfg spec node {type(spec).__name__} at {'.'.join(path)!r}"
    )


def _validate_section_value(
    spec: CfgSectionSpec,
    value: object,
    *,
    path: tuple[str, ...],
) -> None:
    if not isinstance(value, CfgSectionValue):
        _raise_shape_error(path, "CfgSectionValue", type(value).__name__)
    expected_fields = tuple(spec.fields)
    actual_fields = tuple(value.fields)
    if actual_fields != expected_fields:
        _raise_shape_error(
            path,
            f"fields/order {expected_fields!r}",
            f"fields/order {actual_fields!r}",
        )
    for key, node_spec in spec.fields.items():
        _validate_node_value(node_spec, value.fields[key], path=(*path, key))


def _validate_node_value(
    spec: CfgNodeSpec,
    value: CfgNodeValue | None,
    *,
    path: tuple[str, ...],
) -> None:
    if isinstance(spec, LiteralSpec):
        if not isinstance(value, DirectValue) or value.value != spec.value:
            _raise_shape_error(
                path,
                f"DirectValue({spec.value!r})",
                repr(value),
            )
        return
    if isinstance(spec, ScalarSpec):
        if not isinstance(value, (DirectValue, EvalValue)):
            _raise_shape_error(path, "DirectValue or EvalValue", type(value).__name__)
        return
    if isinstance(spec, SweepSpec):
        if not isinstance(value, SweepValue):
            _raise_shape_error(path, "SweepValue", type(value).__name__)
        return
    if isinstance(spec, CenteredSweepSpec):
        if not isinstance(value, CenteredSweepValue):
            _raise_shape_error(path, "CenteredSweepValue", type(value).__name__)
        return
    if isinstance(spec, ReferenceSpec):
        if value is None:
            if not spec.optional:
                _raise_shape_error(path, "ReferenceValue", "None")
            return
        if not isinstance(value, ReferenceValue):
            _raise_shape_error(path, "ReferenceValue", type(value).__name__)
        try:
            selected = select_ref_value_spec(spec, value)
        except RuntimeError as exc:
            allowed = tuple(item.label for item in spec.allowed)
            _raise_shape_error(
                path,
                f"reference matching allowed labels {allowed!r}",
                f"chosen_key {value.chosen_key!r}",
                cause=exc,
            )
        _validate_section_value(selected, value.value, path=path)
        return
    if isinstance(spec, CfgSectionSpec):
        _validate_section_value(spec, value, path=path)
        return
    raise TypeError(
        f"Unsupported cfg spec node {type(spec).__name__} at {'.'.join(path)!r}"
    )


def _raise_shape_error(
    path: tuple[str, ...],
    expected: str,
    actual: str,
    *,
    cause: Exception | None = None,
) -> Never:
    dotted = ".".join(path) or "<root>"
    error = ValueError(
        f"Malformed materialized value at {dotted!r}: expected {expected}; "
        f"actual {actual}"
    )
    if cause is not None:
        raise error from cause
    raise error
