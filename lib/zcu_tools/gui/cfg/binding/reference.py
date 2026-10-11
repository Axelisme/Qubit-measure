from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import cast

from ..inheritance import align_locked_literals, inherit_from, select_ref_value_spec
from ..model import (
    CfgNodeSpec,
    CfgSectionSpec,
    CfgSectionValue,
    ReferenceSpec,
    ReferenceValue,
)
from ..reference_key import (
    is_custom_reference_key,
    make_custom_reference_key,
    parse_custom_reference_key,
)
from .fields import CallbackList, CfgField, SectionField, validate_section_keys
from .ports import ExpressionEvaluator, OptionProvider, ReferenceCatalog

logger = logging.getLogger(__name__)

_ResolvedChoice = tuple[CfgSectionSpec, CfgSectionValue | None]
_UNRESOLVED = object()


class LibraryBindingState(Enum):
    LINKED = "linked"
    CUSTOM = "custom"


@dataclass(frozen=True)
class _PreparedRebuild:
    chosen_key: str
    binding_state: LibraryBindingState
    missing_library_ref: bool
    is_enabled: bool
    sub_field: SectionField | None


def _binding_state_for_key(chosen_key: str) -> LibraryBindingState:
    try:
        if is_custom_reference_key(chosen_key):
            return LibraryBindingState.CUSTOM
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return LibraryBindingState.LINKED


class ReferenceField(CfgField):
    """Reference binding with editable inline and readonly source-key modes.

    spec defines kind, allowed shapes, discriminator and optionality. The supplied
    evaluator, option provider and catalog resolve this fixed binding source.
    initial_val is a ReferenceValue or None (disabled when optional). An empty
    section on a linked key may await its catalog shape; a missing
    entry is Invalid; an unknown shape has no fabricated children. Inline and
    nonempty values require a compatible shape and declared fields. Malformed values
    or failed catalog calls raise rather than falling back.
    """

    spec: ReferenceSpec

    def __init__(
        self,
        spec: ReferenceSpec,
        *,
        evaluate_expression: ExpressionEvaluator,
        provide_options: OptionProvider,
        references: ReferenceCatalog,
        initial_val: object = None,
    ) -> None:
        super().__init__(spec)
        initial_spec: CfgSectionSpec | None = None
        if isinstance(initial_val, ReferenceValue) and not (
            _binding_state_for_key(initial_val.chosen_key) is LibraryBindingState.LINKED
            and not initial_val.value.fields
        ):
            initial_spec = select_ref_value_spec(spec, initial_val)
            validate_section_keys(
                initial_spec,
                initial_val.value,
                path=spec.label or "<reference>",
            )
        self._evaluate_expression = evaluate_expression
        self._provide_options = provide_options
        self._references = references
        self._available_keys = self._load_available_keys()

        if isinstance(initial_val, ReferenceValue):
            self._chosen_key = initial_val.chosen_key
            initial_section: CfgSectionValue | None = initial_val.value
        else:
            first_label = spec.allowed[0].label
            self._chosen_key = make_custom_reference_key(first_label or "Custom")
            initial_section = None

        self._binding_state = _binding_state_for_key(self._chosen_key)
        self.sub_field: SectionField | None = None
        self._missing_library_ref = False
        self._is_enabled = not (spec.optional and initial_val is None)
        self.on_enabled_changed = CallbackList()
        prepared = self._prepare_rebuild(
            chosen_key=self._chosen_key,
            binding_state=self._binding_state,
            is_enabled=self._is_enabled,
            hint=initial_section,
        )
        self._commit_rebuild(prepared)

    @property
    def is_enabled(self) -> bool:
        self._require_open()
        return self._is_enabled

    def available_keys(self) -> tuple[str, ...]:
        self._require_open()
        return self._available_keys

    def available_options(self) -> tuple[str, ...]:
        """Return canonical custom labels and cached library keys for readers."""
        self._require_open()
        return (*(spec.label for spec in self.spec.allowed), *self._available_keys)

    def _load_available_keys(self) -> tuple[str, ...]:
        allowed_labels = frozenset(item.label for item in self.spec.allowed)
        return tuple(self._references.keys(self.spec.kind, allowed_labels))

    def has_missing_library_ref(self) -> bool:
        self._require_open()
        return self._missing_library_ref

    def get_chosen_key(self) -> str:
        self._require_open()
        return self._chosen_key

    def set_chosen_key(self, key: str) -> None:
        self._require_open()
        self._require_writable()
        if key != self._chosen_key:
            resolved = self._resolve_key(key)
            prepared = self._prepare_rebuild(
                chosen_key=key,
                binding_state=_binding_state_for_key(key),
                is_enabled=self._is_enabled,
                chosen_resolution=resolved,
            )
            self._commit_rebuild(prepared)
            self.on_change.emit()

    def set_enabled(self, enabled: bool) -> None:  # noqa: FBT001 - binding API
        self._require_open()
        self._require_writable()
        if not self.spec.optional:
            return
        if enabled != self.is_enabled:
            self._is_enabled = enabled
            self._refresh_validity()
            self.on_enabled_changed.emit(enabled)
            self.on_change.emit()

    def get_value(self) -> ReferenceValue | None:
        self._require_open()
        if self.spec.optional and not self.is_enabled:
            return None
        sub_value = self.sub_field.get_value() if self.sub_field else CfgSectionValue()
        return ReferenceValue(
            chosen_key=self._chosen_key,
            value=sub_value,
            resolved_label=(
                self.sub_field.spec.label
                if self.sub_field is not None and not self._missing_library_ref
                else None
            ),
            error=(
                f"Unknown {self.spec.kind} reference: {self._chosen_key!r}"
                if self._missing_library_ref
                else None
            ),
        )

    def set_value(self, value: object) -> None:
        self._require_writable()
        if value is None:
            self.set_enabled(False)
            return
        if not isinstance(value, ReferenceValue):
            raise TypeError(
                f"ReferenceField expects ReferenceValue or None, got {type(value).__name__}"
            )
        chosen_spec = select_ref_value_spec(self.spec, value)
        validate_section_keys(
            chosen_spec,
            value.value,
            path=self.spec.label or "<reference>",
        )
        binding_state = _binding_state_for_key(value.chosen_key)
        resolved = self._resolve_key(value.chosen_key)
        prepared = self._prepare_rebuild(
            chosen_key=value.chosen_key,
            binding_state=binding_state,
            is_enabled=True,
            hint=value.value,
            chosen_resolution=resolved,
        )
        self._commit_rebuild(prepared)
        self.on_change.emit()

    def refresh_expressions(self) -> None:
        self._require_open()
        if self.sub_field:
            self.sub_field.refresh_expressions()
            self._refresh_validity()

    def refresh_options(self, source_id: str | None = None) -> None:
        self._require_open()
        if self.sub_field:
            self.sub_field.refresh_options(source_id)
            self._refresh_validity()

    def refresh_references(self, kind: str | None = None) -> None:
        self._require_open()
        if self._binding_state is LibraryBindingState.CUSTOM:
            if kind is None or kind == self.spec.kind:
                new_keys = self._load_available_keys()
                changed = new_keys != self._available_keys
                self._available_keys = new_keys
                if changed:
                    self.on_change.emit()
            if self.sub_field:
                self.sub_field.refresh_references(kind)
                self._refresh_validity()
            return
        if kind is None or kind == self.spec.kind:
            self._available_keys = self._load_available_keys()
            prepared = self._prepare_rebuild(
                chosen_key=self._chosen_key,
                binding_state=LibraryBindingState.LINKED,
                is_enabled=self._is_enabled,
            )
            self._commit_rebuild(prepared)
            self.on_change.emit()
            return
        if self.sub_field:
            self.sub_field.refresh_references(kind)
            self._refresh_validity()

    def teardown(self) -> None:
        self.on_enabled_changed.clear()
        if self.sub_field:
            self.sub_field.on_change.disconnect(self._on_sub_change)
            self.sub_field.on_validity_changed.disconnect(self._on_sub_validity_change)
            self.sub_field.teardown()
        super().teardown()

    def _prepare_rebuild(
        self,
        *,
        chosen_key: str,
        binding_state: LibraryBindingState,
        is_enabled: bool,
        hint: CfgSectionValue | None = None,
        chosen_resolution: _ResolvedChoice | None | object = _UNRESOLVED,
    ) -> _PreparedRebuild:
        old_spec = self.sub_field.spec if self.sub_field else None
        old_value = self.sub_field.get_value() if self.sub_field else None
        missing_library_ref = False
        chosen_spec: CfgSectionSpec | None = None
        catalog_value: CfgSectionValue | None = None
        if chosen_resolution is _UNRESOLVED:
            resolved = self._resolve_key(chosen_key)
        else:
            resolved = cast(_ResolvedChoice | None, chosen_resolution)
        if resolved is None:
            missing_library_ref = True
            chosen_spec = self._spec_for_missing_ref_value(
                old_spec, hint, chosen_key=chosen_key
            )
        if resolved is not None:
            chosen_spec, catalog_value = resolved

        replacement: SectionField | None = None
        if chosen_spec is not None:
            if (
                binding_state is LibraryBindingState.LINKED
                and catalog_value is not None
            ):
                value = catalog_value
            elif hint is not None:
                value = hint
            elif catalog_value is not None:
                value = catalog_value
            elif isinstance(old_spec, CfgSectionSpec) and isinstance(
                old_value, CfgSectionValue
            ):
                value = inherit_from(old_value, old_spec, chosen_spec)
            else:
                value = None
            replacement = SectionField(
                chosen_spec,
                evaluate_expression=self._evaluate_expression,
                provide_options=self._provide_options,
                references=self._references,
                initial_val=value,
            )

        if missing_library_ref:
            logger.warning("Reference %r no longer exists in the catalog", chosen_key)

        return _PreparedRebuild(
            chosen_key=chosen_key,
            binding_state=binding_state,
            missing_library_ref=missing_library_ref,
            is_enabled=is_enabled,
            sub_field=replacement,
        )

    def _commit_rebuild(self, prepared: _PreparedRebuild) -> None:
        old_sub_field = self.sub_field
        enabled_changed = prepared.is_enabled != self._is_enabled
        if old_sub_field is not None:
            old_sub_field.on_change.disconnect(self._on_sub_change)
            old_sub_field.on_validity_changed.disconnect(self._on_sub_validity_change)

        self._chosen_key = prepared.chosen_key
        self._binding_state = prepared.binding_state
        self._missing_library_ref = prepared.missing_library_ref
        self._is_enabled = prepared.is_enabled
        self.sub_field = prepared.sub_field
        self.set_readonly(readonly=self.readonly)
        if self.sub_field is not None:
            self.sub_field.on_change.connect(self._on_sub_change)
            self.sub_field.on_validity_changed.connect(self._on_sub_validity_change)
        if old_sub_field is not None:
            old_sub_field.teardown()
        self._refresh_validity()
        if enabled_changed:
            self.on_enabled_changed.emit(prepared.is_enabled)

    def _resolve_key(self, chosen_key: str) -> _ResolvedChoice | None:
        try:
            label = parse_custom_reference_key(chosen_key)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if label is not None:
            for spec in self.spec.allowed:
                if (spec.label or "Custom") == label:
                    return spec, None
            raise RuntimeError(f"Unknown custom reference label: {label!r}")

        resolved = self._references.resolve(self.spec.kind, chosen_key)
        if resolved is None:
            return None
        chosen_spec = next(
            (spec for spec in self.spec.allowed if spec.label == resolved.label),
            None,
        )
        if chosen_spec is None:
            raise RuntimeError(
                f"Reference {chosen_key!r} resolved unsupported shape "
                f"{resolved.label!r} for {self.spec.label!r}"
            )
        if resolved.value is None:
            raise RuntimeError(
                f"Reference catalog cannot materialize supported shape "
                f"{resolved.label!r} for {chosen_key!r}"
            )
        return chosen_spec, align_locked_literals(chosen_spec, resolved.value)

    def _spec_for_missing_ref_value(
        self,
        old_spec: CfgNodeSpec | None,
        value: CfgSectionValue | None,
        *,
        chosen_key: str,
    ) -> CfgSectionSpec | None:
        if isinstance(old_spec, CfgSectionSpec):
            for spec in self.spec.allowed:
                if spec.label == old_spec.label:
                    return old_spec
        if isinstance(value, CfgSectionValue):
            try:
                return select_ref_value_spec(
                    self.spec, ReferenceValue(chosen_key, value)
                )
            except RuntimeError:
                return None
        return None

    def convert_reference_to_inline(self) -> None:
        """Copy this valid ref's published actual values into the same inline shape.

        Nested ref/Eval values freeze recursively, without querying sources.
        Disabled, inline, missing or Invalid values raise CfgInputError INVALID_VALUE.
        A ref ancestor prevents conversion with READONLY.
        """
        from ..resolved import freeze_resolved_value
        from ..resource import CfgInputError, CfgInputReason

        self._require_writable()
        if (
            not self.is_enabled
            or self._binding_state is not LibraryBindingState.LINKED
            or not self.is_valid()
        ):
            raise CfgInputError(
                CfgInputReason.INVALID_VALUE, "conversion requires a valid ref"
            )
        try:
            value = freeze_resolved_value(self.get_value())
        except RuntimeError as exc:
            raise CfgInputError(CfgInputReason.INVALID_VALUE, str(exc)) from exc
        self.set_value(value)

    def set_readonly(self, *, readonly: bool) -> None:
        """Propagate the ancestor lock, retaining this ref mode's own child lock."""
        super().set_readonly(readonly=readonly)
        if self.sub_field is not None:
            self.sub_field.set_readonly(
                readonly=readonly or self._binding_state is LibraryBindingState.LINKED
            )

    def _on_sub_change(self, *_: object) -> None:
        self.on_change.emit()

    def _on_sub_validity_change(self, *_: object) -> None:
        self._refresh_validity()

    def _refresh_validity(self) -> None:
        if self.spec.optional and not self.is_enabled:
            self._set_valid(True)
            return
        if self._missing_library_ref:
            self._set_valid(False)
            return
        self._set_valid(self.sub_field is None or self.sub_field.is_valid())
