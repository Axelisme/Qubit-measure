"""``CfgBuilder`` — fluent assembly of an adapter's default value tree.

An adapter's ``make_default_value(ctx)`` must return a structurally-complete,
spec-compliant value tree (ADR-0010/0011). ``CfgBuilder`` starts from the L1
blank tree, mounts role-backed references through the L2 ``ROLE_FACTORIES``
table, and exposes path-addressed scalar and sweep replacement.

The builder owns domain assembly only. Generic Spec/Value path mechanics live
in ``zcu_tools.gui.cfg``; locking remains a spec-layer decision, and ``build``
only aligns mounted values to those declared literals. Validation stays at the
cfg boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Self, cast

from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarLeafInput,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    align_locked_literals,
    make_default_value,
    read_value_path,
    replace_value_path,
    resolve_spec_path,
    select_ref_value_spec,
)
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ScalarType,
    UnavailableValue,
    ValueRef,
    resolve_value_ref,
)

from .defaults import ROLE_FACTORIES

if TYPE_CHECKING:
    from zcu_tools.gui.session.types import ExpContext


class _NoDefault:
    pass


_NO_DEFAULT = _NoDefault()


class RoleInit(Enum):
    """How ``CfgBuilder.role`` initializes a role-backed reference node."""

    ADOPT = "adopt"
    INLINE = "inline"
    DISABLED = "disabled"


class CfgBuilder:
    """One-shot, path-addressed assembler for an adapter default value tree."""

    def __init__(self, ctx: ExpContext, spec: CfgSectionSpec) -> None:
        self._ctx = ctx
        self._spec = spec
        self._value = make_default_value(spec)
        self._built = False

    def scalars(self, **values: ScalarLeafInput) -> Self:
        """Set top-level scalar leaves by field name."""
        self._check_mutable()
        for key, value in values.items():
            if key not in self._spec.fields:
                raise RuntimeError(
                    f"CfgBuilder.scalars: unknown top-level field {key!r} "
                    f"(available: {', '.join(self._spec.fields)})"
                )
            self._value.with_field(key, value)
        return self

    def set(self, path: str, value: ScalarLeafInput) -> Self:
        """Override an unlocked scalar leaf at ``path``."""
        self._check_mutable()
        leaf_spec = resolve_spec_path(self._spec, path)
        if isinstance(leaf_spec, LiteralSpec):
            raise RuntimeError(
                f"CfgBuilder.set: path {path!r} is a locked literal "
                "(declared via lock_literal in cfg_spec); build() fills it "
                "automatically — do not set it"
            )
        self._value.with_field(path, value)
        return self

    def value_source(
        self,
        path: str,
        key: str,
        *,
        type_name: str | None = None,
        default: ScalarLeafInput | _NoDefault = _NO_DEFAULT,
    ) -> Self:
        """Resolve a registered value source once and store a direct leaf."""
        self._check_mutable()
        target_type = self._value_source_target_type(path)
        try:
            value = resolve_value_ref(
                ValueRef(key, type_name), self._ctx.values, target_type=target_type
            )
        except (MissingValue, UnavailableValue):
            if isinstance(default, _NoDefault):
                raise
            value = default
        return self.set(path, value)

    def role(
        self,
        path: str,
        role_id: str,
        init: RoleInit = RoleInit.ADOPT,
        *,
        blank_overrides: Mapping[str, ScalarLeafInput] | None = None,
    ) -> Self:
        """Mount a role reference and optionally customize an inline blank.

        ``blank_overrides`` uses paths relative to the mounted reference value.
        It applies only when the selected product is a custom reference. Linked
        library values and disabled references remain untouched.
        """
        self._check_mutable()
        if not isinstance(init, RoleInit):
            raise RuntimeError(
                "CfgBuilder.role: init must be a RoleInit value "
                f"(got {type(init).__name__})"
            )
        leaf_spec = resolve_spec_path(self._spec, path)
        if not isinstance(leaf_spec, ReferenceSpec):
            raise RuntimeError(
                f"CfgBuilder.role: spec at {path!r} is "
                f"{type(leaf_spec).__name__}, not a ReferenceSpec"
            )

        role = ROLE_FACTORIES.get(role_id)
        if role is None:
            raise RuntimeError(
                f"CfgBuilder.role: unknown role_id {role_id!r} "
                f"(available: {', '.join(sorted(ROLE_FACTORIES))})"
            )
        self._check_role_kind(leaf_spec, role.kind, path, role_id)

        if init is RoleInit.DISABLED:
            if not leaf_spec.optional:
                raise RuntimeError(
                    f"CfgBuilder.role: RoleInit.DISABLED at {path!r} but the "
                    "spec ref is not optional (a required ref cannot be disabled)"
                )
            if role.ref is None:
                raise RuntimeError(
                    f"CfgBuilder.role: role {role_id!r} has no library-aware "
                    "(ref) factory, so RoleInit.DISABLED (library-miss → None) "
                    "is unsupported"
                )
            node = role.ref(self._ctx, optional=True)
        elif init is RoleInit.INLINE or role.ref is None:
            node = role.blank(self._ctx)
        else:
            node = role.ref(self._ctx)

        self._check_ref_value(node, path, role_id)
        self._apply_blank_overrides(leaf_spec, node, blank_overrides, path)
        replace_value_path(self._value, path, node)
        return self

    def sweep(self, path: str, value: SweepValue) -> Self:
        """Mount a pre-built ``SweepValue`` at ``path``."""
        self._check_mutable()
        if not isinstance(value, SweepValue):
            raise RuntimeError(
                "CfgBuilder.sweep: value must be a SweepValue "
                f"(got {type(value).__name__})"
            )
        leaf_spec = resolve_spec_path(self._spec, path)
        if not isinstance(leaf_spec, SweepSpec):
            raise RuntimeError(
                f"CfgBuilder.sweep: spec at {path!r} is "
                f"{type(leaf_spec).__name__}, not a SweepSpec"
            )
        replace_value_path(self._value, path, value)
        return self

    def build(self) -> CfgSectionValue:
        """Align locked literals and return the assembled value tree."""
        self._check_mutable()
        align_locked_literals(self._spec, self._value)
        self._built = True
        return self._value

    def _check_mutable(self) -> None:
        if self._built:
            raise RuntimeError("CfgBuilder is already built; create a new one")

    def _value_source_target_type(self, path: str) -> ScalarType:
        leaf_spec = resolve_spec_path(self._spec, path)
        if isinstance(leaf_spec, ScalarSpec):
            if leaf_spec.type not in (int, float, str, bool):
                raise RuntimeError(
                    f"CfgBuilder.value_source: scalar field at {path!r} has "
                    f"unsupported type {leaf_spec.type.__name__!r}"
                )
            return cast(ScalarType, leaf_spec.type)
        raise RuntimeError(
            f"CfgBuilder.value_source: spec at {path!r} is "
            f"{type(leaf_spec).__name__}, not a scalar leaf"
        )

    @staticmethod
    def _apply_blank_overrides(
        ref_spec: ReferenceSpec,
        node: ReferenceValue | None,
        overrides: Mapping[str, ScalarLeafInput] | None,
        path: str,
    ) -> None:
        if not overrides or node is None or not node.chosen_key.startswith("<Custom:"):
            return

        value_spec = select_ref_value_spec(ref_spec, node)
        for relative_path in overrides:
            leaf_spec = resolve_spec_path(value_spec, relative_path)
            if isinstance(leaf_spec, LiteralSpec):
                raise RuntimeError(
                    f"CfgBuilder.role: blank override {relative_path!r} at "
                    f"{path!r} is a locked literal"
                )
            if not isinstance(leaf_spec, ScalarSpec):
                raise RuntimeError(
                    f"CfgBuilder.role: blank override {relative_path!r} at "
                    f"{path!r} targets {type(leaf_spec).__name__}, not a scalar leaf"
                )
            read_value_path(node.value, relative_path)

        for relative_path, value in overrides.items():
            node.value.with_field(relative_path, value)

    @staticmethod
    def _check_role_kind(
        ref_spec: ReferenceSpec,
        role_kind: str,
        path: str,
        role_id: str,
    ) -> None:
        if ref_spec.kind != role_kind:
            raise RuntimeError(
                f"CfgBuilder.role: spec at {path!r} expects reference kind "
                f"{ref_spec.kind!r}, but role {role_id!r} has kind {role_kind!r}"
            )

    @staticmethod
    def _check_ref_value(
        node: ReferenceValue | None,
        path: str,
        role_id: str,
    ) -> None:
        if node is None:
            return
        if not isinstance(node, ReferenceValue):
            raise RuntimeError(
                f"CfgBuilder.role: role {role_id!r} at {path!r} produced "
                f"{type(node).__name__}, not ReferenceValue"
            )
