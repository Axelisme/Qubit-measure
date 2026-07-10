"""Private paired Spec/Value tree mechanics for node schema declaration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

from zcu_tools.gui.cfg import (
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
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    align_locked_literals,
    make_default_value,
)


class _SchemaTree:
    """Declare an initially empty Spec/Value tree in lockstep."""

    def __init__(self, *, label: str) -> None:
        self._spec = CfgSectionSpec(label=label, fields={})
        self._value = CfgSectionValue(fields={})
        self._choice_bindings: list[
            tuple[str, str, Mapping[str, tuple[str, ...]], str | None]
        ] = []

    @property
    def spec(self) -> CfgSectionSpec:
        return self._spec

    def declare(self, path: str, spec: CfgNodeSpec, default: object) -> None:
        value = _wrap_default(spec, default)
        parent_spec, parent_value, leaf = _ensure_parent(self._spec, self._value, path)
        if leaf in parent_spec.fields:
            raise ValueError(f"cfg path {path!r} already exists")
        parent_spec.fields[leaf] = spec
        parent_value.fields[leaf] = value

    def ensure_section(self, path: str, label: str) -> None:
        parent_spec, parent_value, leaf = _ensure_parent(self._spec, self._value, path)
        spec = parent_spec.fields.get(leaf)
        value = parent_value.fields.get(leaf)
        if spec is None:
            parent_spec.fields[leaf] = CfgSectionSpec(label=label, fields={})
            parent_value.fields[leaf] = CfgSectionValue(fields={})
            return
        if not isinstance(spec, CfgSectionSpec) or not isinstance(
            value, CfgSectionValue
        ):
            raise TypeError(f"cfg path {path!r} is not a section")
        if label and not spec.label:
            parent_spec.fields[leaf] = replace(spec, label=label)

    def validate_declarations(self, paths: tuple[str, ...]) -> None:
        """Check a declaration batch without modifying the paired tree."""
        spec = deepcopy(self._spec)
        value = deepcopy(self._value)
        for path in paths:
            parent_spec, _, leaf = _ensure_parent(spec, value, path)
            if leaf in parent_spec.fields:
                raise ValueError(f"cfg path {path!r} already exists")

    def add_choice_binding(
        self,
        section_path: str,
        selector_key: str,
        fields_by_choice: Mapping[str, tuple[str, ...]],
        *,
        section_label: str | None,
    ) -> None:
        if not fields_by_choice:
            raise ValueError("choice_fields needs at least one choice")
        self._choice_bindings.append(
            (section_path, selector_key, fields_by_choice, section_label)
        )

    def build(self) -> CfgSchema:
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
        return CfgSchema(spec=spec, value=value)


def _default_value_for_spec(spec: CfgNodeSpec) -> CfgNodeValue | None:
    return make_default_value(CfgSectionSpec(fields={"value": spec})).fields["value"]


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
) -> tuple[CfgSectionSpec, CfgSectionValue, str]:
    parts = _split_path(path)
    spec = root_spec
    value = root_value
    for index, part in enumerate(parts[:-1]):
        child_spec = spec.fields.get(part)
        child_value = value.fields.get(part)
        if child_spec is None:
            child_spec = CfgSectionSpec(label=_section_label(part), fields={})
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


def _section_label(key: str) -> str:
    labels = {
        "modules": "Modules",
        "sweep": "Sweep",
        "generation": "Generation overrides",
        "acquisition": "Acquisition guardrails",
        "relax": "Relax timing",
    }
    return labels.get(key, key.replace("_", " ").title())
