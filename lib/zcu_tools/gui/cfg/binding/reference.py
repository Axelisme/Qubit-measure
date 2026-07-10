from __future__ import annotations

import logging
from enum import Enum

from ..inheritance import align_locked_literals, inherit_from, select_ref_value_spec
from ..model import (
    CfgNodeSpec,
    CfgSectionSpec,
    CfgSectionValue,
    ReferenceSpec,
    ReferenceValue,
)
from .fields import CallbackList, CfgField, SectionField
from .ports import ExpressionEvaluator, OptionProvider, ReferenceCatalog

logger = logging.getLogger(__name__)


class LibraryBindingState(Enum):
    LINKED = "linked"
    MODIFIED = "modified"
    CUSTOM = "custom"


def _binding_state_for_key(chosen_key: str) -> LibraryBindingState:
    if chosen_key.startswith("<Custom:"):
        return LibraryBindingState.CUSTOM
    return LibraryBindingState.LINKED


class ReferenceField(CfgField):
    """Reference binding with recoverable linked, modified, and missing states."""

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
        self._evaluate_expression = evaluate_expression
        self._provide_options = provide_options
        self._references = references
        self._available_keys = self._load_available_keys()

        init_overridden = False
        if isinstance(initial_val, ReferenceValue):
            self._chosen_key = initial_val.chosen_key
            initial_section: CfgSectionValue | None = initial_val.value
            init_overridden = initial_val.is_overridden
        else:
            first_label = spec.allowed[0].label
            self._chosen_key = f"<Custom:{first_label}>"
            initial_section = None

        self._binding_state = _binding_state_for_key(self._chosen_key)
        if init_overridden and self._binding_state is LibraryBindingState.LINKED:
            self._binding_state = LibraryBindingState.MODIFIED
        self.sub_field: SectionField | None = None
        self._missing_library_ref = False
        self._is_enabled = not (spec.optional and initial_val is None)
        self.on_enabled_changed = CallbackList()
        self._rebuild_sub_field(hint=initial_section)

    @property
    def is_enabled(self) -> bool:
        self._require_open()
        return self._is_enabled

    def available_keys(self) -> tuple[str, ...]:
        self._require_open()
        return self._available_keys

    def _load_available_keys(self) -> tuple[str, ...]:
        allowed_labels = frozenset(item.label for item in self.spec.allowed)
        return tuple(self._references.keys(self.spec.kind, allowed_labels))

    def is_modified(self) -> bool:
        self._require_open()
        return self._binding_state is LibraryBindingState.MODIFIED

    def has_missing_library_ref(self) -> bool:
        self._require_open()
        return self._missing_library_ref

    def get_chosen_key(self) -> str:
        self._require_open()
        return self._chosen_key

    def set_chosen_key(self, key: str) -> None:
        self._require_open()
        if key != self._chosen_key or self.is_modified():
            self._resolve_key(key)
            self._chosen_key = key
            self._binding_state = _binding_state_for_key(key)
            self._rebuild_sub_field()
            self.on_change.emit(self.get_value())

    def set_enabled(self, enabled: bool) -> None:
        self._require_open()
        if not self.spec.optional:
            return
        if enabled != self.is_enabled:
            self._is_enabled = enabled
            self._refresh_validity()
            self.on_enabled_changed.emit(enabled)
            self.on_change.emit(self.get_value())

    def get_value(self) -> ReferenceValue | None:
        self._require_open()
        if self.spec.optional and not self.is_enabled:
            return None
        sub_value = self.sub_field.get_value() if self.sub_field else CfgSectionValue()
        return ReferenceValue(
            chosen_key=self._chosen_key,
            value=sub_value,
            is_overridden=self.is_modified(),
        )

    def set_value(self, value: object) -> None:
        self._require_open()
        if value is None:
            self.set_enabled(False)
            return
        if not isinstance(value, ReferenceValue):
            raise TypeError(
                f"ReferenceField expects ReferenceValue or None, got {type(value).__name__}"
            )
        if self.spec.optional and not self.is_enabled:
            self.set_enabled(True)
        if value.chosen_key != self._chosen_key or self.is_modified():
            self._resolve_key(value.chosen_key)
            self._chosen_key = value.chosen_key
            self._binding_state = _binding_state_for_key(value.chosen_key)
            if (
                value.is_overridden
                and self._binding_state is LibraryBindingState.LINKED
            ):
                self._binding_state = LibraryBindingState.MODIFIED
            self._rebuild_sub_field(hint=value.value)
        elif self.sub_field:
            self.sub_field.set_value(value.value)
        self.on_change.emit(self.get_value())

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
        if kind is None or kind == self.spec.kind:
            new_keys = self._load_available_keys()
            keys_changed = new_keys != self._available_keys
            self._available_keys = new_keys
            binding_emitted = self._refresh_catalog_binding()
            if self._binding_state is LibraryBindingState.CUSTOM and self.sub_field:
                self.sub_field.refresh_references(kind)
                self._refresh_validity()
            if keys_changed and not binding_emitted:
                self.on_change.emit(self.get_value())
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

    def _rebuild_sub_field(self, hint: CfgSectionValue | None = None) -> None:
        old_spec = self.sub_field.spec if self.sub_field else None
        old_value = self.sub_field.get_value() if self.sub_field else None
        self._missing_library_ref = False
        chosen_spec: CfgSectionSpec | None = None
        catalog_value: CfgSectionValue | None = None
        resolved = self._resolve_chosen()
        if resolved is None:
            if (
                self._binding_state is LibraryBindingState.MODIFIED
                and old_spec is not None
            ):
                kept_value = old_value if old_value is not None else hint
                label = self._custom_label_for_value(old_spec, kept_value)
                logger.warning(
                    "Reference %r (modified) no longer exists; converting to "
                    "inline <Custom:%s>, keeping edits",
                    self._chosen_key,
                    label,
                )
                self._chosen_key = f"<Custom:{label}>"
                self._binding_state = LibraryBindingState.CUSTOM
                resolved = self._resolve_chosen()
                if resolved is None:
                    raise RuntimeError(
                        "Custom reference resolution unexpectedly returned missing"
                    )
                if hint is None:
                    hint = kept_value
            else:
                logger.warning(
                    "Reference %r no longer exists in the catalog",
                    self._chosen_key,
                )
                self._missing_library_ref = True
                chosen_spec = self._spec_for_missing_ref_value(old_spec, hint)
                catalog_value = None
        if resolved is not None:
            chosen_spec, catalog_value = resolved

        if self.sub_field:
            self.sub_field.on_change.disconnect(self._on_sub_change)
            self.sub_field.on_validity_changed.disconnect(self._on_sub_validity_change)
            self.sub_field.teardown()
            self.sub_field = None

        if chosen_spec is None:
            self._refresh_validity()
            return
        if hint is not None:
            value = hint
        elif catalog_value is not None:
            value = catalog_value
        elif isinstance(old_spec, CfgSectionSpec) and isinstance(
            old_value, CfgSectionValue
        ):
            value = inherit_from(old_value, old_spec, chosen_spec)
        else:
            value = None
        self.sub_field = SectionField(
            chosen_spec,
            evaluate_expression=self._evaluate_expression,
            provide_options=self._provide_options,
            references=self._references,
            initial_val=value,
        )
        self.sub_field.on_change.connect(self._on_sub_change)
        self.sub_field.on_validity_changed.connect(self._on_sub_validity_change)
        self._refresh_validity()

    def _resolve_chosen(
        self,
    ) -> tuple[CfgSectionSpec, CfgSectionValue | None] | None:
        return self._resolve_key(self._chosen_key)

    def _resolve_key(
        self, chosen_key: str
    ) -> tuple[CfgSectionSpec, CfgSectionValue | None] | None:
        if chosen_key.startswith("<Custom:"):
            if not chosen_key.endswith(">"):
                raise RuntimeError(f"Invalid custom reference key: {chosen_key!r}")
            label = chosen_key[len("<Custom:") : -1]
            for spec in self.spec.allowed:
                if spec.label == label:
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
    ) -> CfgSectionSpec | None:
        if isinstance(old_spec, CfgSectionSpec):
            for spec in self.spec.allowed:
                if spec.label == old_spec.label:
                    return old_spec
        if isinstance(value, CfgSectionValue):
            try:
                return select_ref_value_spec(
                    self.spec, ReferenceValue(self._chosen_key, value)
                )
            except RuntimeError:
                return None
        return None

    def _custom_label_for_value(
        self,
        old_spec: CfgNodeSpec | None,
        value: CfgSectionValue | None,
    ) -> str:
        if isinstance(old_spec, CfgSectionSpec) and any(
            spec.label == old_spec.label for spec in self.spec.allowed
        ):
            return old_spec.label
        if isinstance(value, CfgSectionValue):
            try:
                return select_ref_value_spec(
                    self.spec, ReferenceValue(self._chosen_key, value)
                ).label
            except RuntimeError:
                pass
        return self.spec.allowed[0].label

    def _on_sub_change(self, *_: object) -> None:
        if self._binding_state is LibraryBindingState.LINKED:
            self._binding_state = LibraryBindingState.MODIFIED
        self.on_change.emit(self.get_value())

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

    def _refresh_catalog_binding(self) -> bool:
        if self._binding_state is LibraryBindingState.CUSTOM:
            return False
        if self._missing_library_ref:
            previous_state = self._binding_state
            self._binding_state = LibraryBindingState.LINKED
            self._rebuild_sub_field()
            if self._missing_library_ref:
                self._binding_state = previous_state
            self.on_change.emit(self.get_value())
            return True
        if (
            self._catalog_key_present()
            and self._binding_state is not LibraryBindingState.LINKED
        ):
            return False
        self._rebuild_sub_field()
        self.on_change.emit(self.get_value())
        return True

    def _catalog_key_present(self) -> bool:
        return self._references.resolve(self.spec.kind, self._chosen_key) is not None
