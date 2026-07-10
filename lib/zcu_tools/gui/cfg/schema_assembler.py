"""Domain-free lockstep assembly of configuration Spec and Value trees."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Final, Self

from .inheritance import align_locked_literals, make_default_value
from .model import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ChoiceBinding,
    ChoiceSectionSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)


class _UseSpecDefault:
    pass


USE_SPEC_DEFAULT: Final = _UseSpecDefault()
"""Ask :class:`CfgSchemaAssembler` for the spec's neutral default value."""


def default_value_for_spec(spec: CfgNodeSpec) -> CfgNodeValue | None:
    """Return the neutral value for one spec node.

    This is intentionally structural: it knows no experiment roles or domain
    defaults. Domain builders may use it while defining a static shape whose
    real default will only be materialized later.
    """

    return make_default_value(CfgSectionSpec(fields={"value": spec})).fields["value"]


class CfgSchemaAssembler:
    """Declare an initially empty Spec/Value tree in lockstep.

    The assembler owns only paired-tree mechanics. Section labels may be
    supplied by a consumer-owned callback; no application vocabulary lives in
    this module.
    """

    def __init__(
        self,
        *,
        label: str = "",
        section_labeler: Callable[[str], str] | None = None,
    ) -> None:
        self._spec = CfgSectionSpec(label=label, fields={})
        self._value = CfgSectionValue(fields={})
        self._section_labeler = section_labeler or _generic_section_label
        self._choice_bindings: list[
            tuple[str, str, Mapping[str, tuple[str, ...]], str | None]
        ] = []
        self._built = False

    @property
    def spec(self) -> CfgSectionSpec:
        """An isolated snapshot of the in-progress spec for preflight."""

        return deepcopy(self._spec)

    @staticmethod
    def default_value_for_spec(spec: CfgNodeSpec) -> CfgNodeValue | None:
        return default_value_for_spec(spec)

    def declare(
        self,
        path: str,
        spec: CfgNodeSpec,
        default: object = USE_SPEC_DEFAULT,
    ) -> Self:
        self._check_mutable()
        owned_spec = deepcopy(spec)
        resolved_default = (
            default_value_for_spec(owned_spec)
            if default is USE_SPEC_DEFAULT
            else default
        )
        owned_value = deepcopy(_wrap_default(owned_spec, resolved_default))
        parent_spec, parent_value, leaf = _ensure_parent(
            self._spec,
            self._value,
            path,
            section_labeler=self._section_labeler,
        )
        if leaf in parent_spec.fields:
            raise ValueError(f"cfg path {path!r} already exists")
        parent_spec.fields[leaf] = owned_spec
        parent_value.fields[leaf] = owned_value
        return self

    def ensure_section(self, path: str, *, label: str) -> Self:
        self._check_mutable()
        parent_spec, parent_value, leaf = _ensure_parent(
            self._spec,
            self._value,
            path,
            section_labeler=self._section_labeler,
        )
        spec = parent_spec.fields.get(leaf)
        value = parent_value.fields.get(leaf)
        if spec is None:
            parent_spec.fields[leaf] = CfgSectionSpec(label=label, fields={})
            parent_value.fields[leaf] = CfgSectionValue(fields={})
            return self
        if not isinstance(spec, CfgSectionSpec) or not isinstance(
            value, CfgSectionValue
        ):
            raise TypeError(f"cfg path {path!r} is not a section")
        if label and not spec.label:
            parent_spec.fields[leaf] = replace(spec, label=label)
        return self

    def validate_declarations(self, paths: tuple[str, ...]) -> None:
        """Check a declaration batch without modifying the paired tree."""

        self._check_mutable()
        spec = deepcopy(self._spec)
        value = deepcopy(self._value)
        for path in paths:
            parent_spec, parent_value, leaf = _ensure_parent(
                spec,
                value,
                path,
                section_labeler=self._section_labeler,
            )
            if leaf in parent_spec.fields:
                raise ValueError(f"cfg path {path!r} already exists")
            # A concrete placeholder makes duplicates and ancestor/descendant
            # conflicts within the same batch visible during this preflight.
            parent_spec.fields[leaf] = ScalarSpec(label="", type=bool)
            parent_value.fields[leaf] = DirectValue(False)

    def add_choice_binding(
        self,
        section_path: str,
        selector_key: str,
        fields_by_choice: Mapping[str, tuple[str, ...]],
        *,
        section_label: str | None,
    ) -> Self:
        self._check_mutable()
        if not fields_by_choice:
            raise ValueError("choice_fields needs at least one choice")
        self._choice_bindings.append(
            (section_path, selector_key, dict(fields_by_choice), section_label)
        )
        return self

    def build(self) -> CfgSchema:
        """Return an isolated schema snapshot and consume this assembler."""

        self._check_mutable()
        spec = deepcopy(self._spec)
        value = deepcopy(self._value)
        for (
            section_path,
            selector_key,
            fields_by_choice,
            section_label,
        ) in self._choice_bindings:
            _replace_with_choice_section(
                spec,
                section_path,
                selector_key,
                fields_by_choice,
                section_label=section_label,
            )
        align_locked_literals(spec, value)
        self._built = True
        return CfgSchema(spec=spec, value=value)

    def _check_mutable(self) -> None:
        if self._built:
            raise RuntimeError("CfgSchemaAssembler is already built; create a new one")


def _wrap_default(spec: CfgNodeSpec, default: object) -> CfgNodeValue | None:
    if isinstance(spec, SweepSpec):
        if not isinstance(default, SweepValue):
            raise TypeError(
                f"SweepSpec default must be SweepValue, got {type(default).__name__}"
            )
        return default
    if isinstance(spec, CenteredSweepSpec):
        if not isinstance(default, CenteredSweepValue):
            raise TypeError(
                "CenteredSweepSpec default must be CenteredSweepValue, "
                f"got {type(default).__name__}"
            )
        return default
    if isinstance(spec, ScalarSpec):
        if isinstance(default, (DirectValue, EvalValue)):
            return default
        return DirectValue(default)
    if isinstance(spec, LiteralSpec):
        if isinstance(default, DirectValue):
            return default
        return DirectValue(default)
    if isinstance(spec, ReferenceSpec):
        if default is None and spec.optional:
            return None
        if not isinstance(default, ReferenceValue):
            raise TypeError(
                "ReferenceSpec default must be ReferenceValue, "
                f"got {type(default).__name__}"
            )
        return default
    if isinstance(spec, CfgSectionSpec):
        if not isinstance(default, CfgSectionValue):
            raise TypeError(
                "CfgSectionSpec default must be CfgSectionValue, "
                f"got {type(default).__name__}"
            )
        return default
    if default is None:
        return None
    if not isinstance(
        default,
        (
            CenteredSweepValue,
            CfgSectionValue,
            DirectValue,
            EvalValue,
            ReferenceValue,
            SweepValue,
        ),
    ):
        raise TypeError(
            f"unsupported default {type(default).__name__} for {type(spec).__name__}"
        )
    return default


def _ensure_parent(
    root_spec: CfgSectionSpec,
    root_value: CfgSectionValue,
    path: str,
    *,
    section_labeler: Callable[[str], str],
) -> tuple[CfgSectionSpec, CfgSectionValue, str]:
    parts = _split_path(path)
    spec = root_spec
    value = root_value
    for index, part in enumerate(parts[:-1]):
        child_spec = spec.fields.get(part)
        child_value = value.fields.get(part)
        if child_spec is None:
            child_spec = CfgSectionSpec(label=section_labeler(part), fields={})
            child_value = CfgSectionValue(fields={})
            spec.fields[part] = child_spec
            value.fields[part] = child_value
        if not isinstance(child_spec, CfgSectionSpec) or not isinstance(
            child_value, CfgSectionValue
        ):
            parent = ".".join(parts[: index + 1])
            raise TypeError(f"cfg path {path!r} cannot descend through {parent!r}")
        spec = child_spec
        value = child_value
    return spec, value, parts[-1]


def _replace_with_choice_section(
    root_spec: CfgSectionSpec,
    section_path: str,
    selector_key: str,
    fields_by_choice: Mapping[str, tuple[str, ...]],
    *,
    section_label: str | None,
) -> None:
    parent, leaf = _section_parent(root_spec, section_path)
    section = parent.fields[leaf]
    if not isinstance(section, CfgSectionSpec):
        raise TypeError(f"choice section {section_path!r} is not a section")
    binding = _choice_binding(section, selector_key, fields_by_choice)
    bindings = section.bindings if isinstance(section, ChoiceSectionSpec) else ()
    parent.fields[leaf] = ChoiceSectionSpec(
        fields=dict(section.fields),
        label=section.label if section_label is None else section_label,
        inherit_hook=section.inherit_hook,
        bindings=(*bindings, binding),
    )


def _choice_binding(
    section: CfgSectionSpec,
    selector_key: str,
    fields_by_choice: Mapping[str, tuple[str, ...]],
) -> ChoiceBinding:
    variants: dict[str, CfgSectionSpec] = {}
    for value, field_keys in fields_by_choice.items():
        fields: dict[str, CfgNodeSpec] = {}
        for key in field_keys:
            try:
                fields[key] = section.fields[key]
            except KeyError as exc:
                raise ValueError(
                    f"choice {selector_key}={value!r} references unknown field {key!r}"
                ) from exc
        variants[value] = CfgSectionSpec(fields=fields, label=value)
    return ChoiceBinding(selector_key, variants)


def _section_parent(root_spec: CfgSectionSpec, path: str) -> tuple[CfgSectionSpec, str]:
    parts = _split_path(path)
    section = root_spec
    for part in parts[:-1]:
        child = section.fields.get(part)
        if not isinstance(child, CfgSectionSpec):
            raise TypeError(f"cfg path {path!r} cannot descend through {part!r}")
        section = child
    leaf = parts[-1]
    if leaf not in section.fields:
        raise KeyError(f"cfg path {path!r} is missing")
    return section, leaf


def _split_path(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split(".") if part)
    if not parts:
        raise ValueError("cfg path must not be empty")
    return parts


def _generic_section_label(key: str) -> str:
    return key.replace("_", " ").title()


__all__ = ["CfgSchemaAssembler", "USE_SPEC_DEFAULT", "default_value_for_spec"]
