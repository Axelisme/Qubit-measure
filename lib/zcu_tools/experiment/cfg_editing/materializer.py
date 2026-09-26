"""Program-domain policy for raw module/waveform Spec/Value materialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from zcu_tools.gui.cfg import (
    RAW_MISSING,
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    LiteralSpec,
    RawMissing,
    ReferenceMaterialization,
    ReferenceSpec,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    make_custom_reference_key,
    make_default_value,
    materialize_spec_value,
)

from .catalog import (
    PROGRAM_SHAPES,
    ProgramCfgKind,
    ProgramShape,
    ProgramSpecPolicy,
    UnknownProgramShapeError,
)


@dataclass(frozen=True, slots=True)
class ProgramMaterializationPolicy:
    """App-selected materializable subset plus program missing/default rules."""

    spec_policy: ProgramSpecPolicy
    allowed_module_discriminators: frozenset[str]
    allowed_waveform_styles: frozenset[str]

    def scalar_value(
        self,
        path: tuple[str, ...],
        spec: ScalarSpec,
        raw: object | RawMissing,
    ) -> ScalarValue:
        del spec
        if raw is not RAW_MISSING:
            return DirectValue(raw)
        return DirectValue(0 if path[-1] in {"ch", "ro_ch"} else None)

    def sweep_value(
        self,
        path: tuple[str, ...],
        spec: SweepSpec,
        raw: object | RawMissing,
    ) -> SweepValue:
        del spec, raw
        raise TypeError(
            f"Program cfg materialization does not support SweepSpec at "
            f"{'.'.join(path)!r}"
        )

    def centered_sweep_value(
        self,
        path: tuple[str, ...],
        spec: CenteredSweepSpec,
        raw: object | RawMissing,
    ) -> CenteredSweepValue:
        del spec, raw
        raise TypeError(
            f"Program cfg materialization does not support CenteredSweepSpec at "
            f"{'.'.join(path)!r}"
        )

    def missing_section_value(
        self,
        path: tuple[str, ...],
        spec: CfgSectionSpec,
        raw: object | RawMissing,
    ) -> CfgSectionValue:
        del path, raw
        return make_default_value(spec)

    def reference_value(
        self,
        path: tuple[str, ...],
        spec: ReferenceSpec,
        raw: object | RawMissing,
    ) -> ReferenceMaterialization | None:
        if raw is RAW_MISSING or not isinstance(raw, Mapping):
            if spec.optional:
                return None
            selected = spec.allowed[0]
            selected_raw: object | RawMissing = RAW_MISSING
        else:
            discriminator_key = "type" if spec.kind == "module" else "style"
            if discriminator_key not in raw and len(spec.allowed) == 1:
                selected = spec.allowed[0]
            else:
                shape = _shape_from_raw(spec.kind, raw)
                selected = _select_allowed_reference_shape(spec, shape, path)
            selected_raw = raw
        return ReferenceMaterialization(
            spec=selected,
            raw=selected_raw,
            chosen_key=make_custom_reference_key(selected.label or "Custom"),
        )


def materialize_program_module(
    raw: Mapping[str, object],
    policy: ProgramMaterializationPolicy,
) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Materialize one module from the policy's legal app subset."""

    shape = _shape_from_raw("module", raw)
    if shape.discriminator not in policy.allowed_module_discriminators:
        raise RuntimeError(f"Unsupported module type {shape.discriminator!r}")
    if shape.discriminator == "reset/bath" and "relax_delay" in raw:
        raise RuntimeError(
            "Module 'reset/bath' does not accept 'relax_delay'; shot relaxation "
            "belongs to the program root"
        )
    spec = shape.make_spec(policy.spec_policy)
    return spec, materialize_spec_value(spec, raw, policy=policy)


def materialize_program_waveform(
    raw: Mapping[str, object],
    policy: ProgramMaterializationPolicy,
) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Materialize one waveform; a missing root style means Const."""

    shape = _shape_from_raw("waveform", raw)
    if shape.discriminator not in policy.allowed_waveform_styles:
        raise RuntimeError(f"Unsupported waveform style {shape.discriminator!r}")
    spec = shape.make_spec(policy.spec_policy)
    return spec, materialize_spec_value(spec, raw, policy=policy)


def _shape_from_raw(
    kind: str,
    raw: Mapping[str, object],
) -> ProgramShape:
    if kind == "module":
        discriminator = _read_discriminator(raw, "type", missing="")
        message = "Unsupported module type"
        catalog_kind: ProgramCfgKind = "module"
    elif kind == "waveform":
        discriminator = _read_discriminator(raw, "style", missing="const")
        message = "Unsupported waveform style"
        catalog_kind = "waveform"
    else:
        raise RuntimeError(f"Unsupported program reference kind {kind!r}")
    try:
        return PROGRAM_SHAPES.get(catalog_kind, discriminator)
    except UnknownProgramShapeError as exc:
        raise RuntimeError(f"{message} {discriminator!r}") from exc


def _read_discriminator(
    raw: Mapping[str, object],
    key: str,
    *,
    missing: str,
) -> str:
    if key not in raw:
        return missing
    value = raw[key]
    if not isinstance(value, str):
        raise TypeError(
            f"Program discriminator {key!r} must be str, got {type(value).__name__}"
        )
    return value


def _select_allowed_reference_shape(
    spec: ReferenceSpec,
    shape: ProgramShape,
    path: tuple[str, ...],
) -> CfgSectionSpec:
    discriminator_key = "type" if spec.kind == "module" else "style"
    for allowed in spec.allowed:
        literal = allowed.fields.get(discriminator_key)
        if isinstance(literal, LiteralSpec) and literal.value == shape.discriminator:
            return allowed
    allowed = ", ".join(
        repr(cast(LiteralSpec, item.fields[discriminator_key]).value)
        for item in spec.allowed
    )
    raise RuntimeError(
        f"Program reference {'.'.join(path)!r} does not allow {spec.kind} shape "
        f"{shape.discriminator!r}; allowed: {allowed}"
    )
